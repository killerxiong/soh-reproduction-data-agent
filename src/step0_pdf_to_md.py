import io
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import MINERU_BASE_URL, MINERU_TOKEN as ENV_MINERU_TOKEN


# ============================================================
# PyCharm 直接运行配置区
# ============================================================
# 安全原则：
# 1. 本脚本不会删除 OUT_DIR 里的任何已有文件。
# 2. 本脚本不会删除 raw_data/paper1 里的任何内容。
# 3. MinerU zip / images / full.md 只放在系统临时目录，用完自动清理。
# 4. 最终只向 OUT_DIR 写入：
#    - paper.md
#    - Supplementary information.md
#    - step0_manifest.json

PDF_DIR = ""

PAPER_PDF = ""
SUPP_PDF = ""

# 强烈建议输出到独立目录，不要直接输出到 raw_data/paper1。
OUT_DIR = "outputs/demo/_work"

# 推荐不要把 token 提交到 Git。可以直接填字符串，也可以留空后用环境变量 MINERU_TOKEN。
MINERU_TOKEN = ENV_MINERU_TOKEN

MODEL_VERSION = "vlm"
LANGUAGE = "en"
ENABLE_FORMULA = True
ENABLE_TABLE = True

# True: 重新调用 MinerU 并覆盖 OUT_DIR 下的 paper.md / Supplementary information.md。
# False: 如果目标 md 已存在，则跳过。
OVERWRITE = True

TIMEOUT_SECONDS = 3600
POLL_INTERVAL_SECONDS = 10

# 清洗成更接近网页下载版：删除 <details>...</details> 图像理解块。
CLEAN_WEB_LIKE = True

# 因为最终不保存 images 文件夹，所以建议 False，避免 md 里出现失效图片路径。
KEEP_IMAGE_LINES = False


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


def normalize_name(name: str) -> str:
    return Path(str(name)).name.strip().lower()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_mineru_md_like_web(text: str, keep_image_lines: bool = False) -> str:
    """
    清洗 MinerU API 输出，使其更接近网页下载版，同时避免 STEP1 被图像伪表格污染。

    删除：
      1. <details>...</details> 图像理解块
      2. 可选删除图片引用行 ![](images/xxx.jpg)
      3. 多余空行
    """
    text = re.sub(
        r"<details>.*?</details>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not keep_image_lines:
        text = re.sub(
            r"^\s*!\[[^\]]*\]\([^)]+\)\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def check_pdf(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"{name} is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise RuntimeError(f"{name} is not a PDF: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{name} is empty: {path}")


def pick_pdf_files(
    pdf_dir: Optional[Path],
    paper_pdf: Optional[Path],
    supp_pdf: Optional[Path],
) -> Tuple[Path, Optional[Path], List[str]]:
    warnings: List[str] = []

    if paper_pdf is not None:
        paper = Path(paper_pdf)
    else:
        if pdf_dir is None:
            raise RuntimeError("Need PDF_DIR or PAPER_PDF.")
        pdfs = sorted(
            [p for p in Path(pdf_dir).iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
            key=lambda p: natural_key(p.name),
        )
        exact = [p for p in pdfs if p.name.lower() in {"paper.PDF", "main.PDF", "article.PDF"}]
        if exact:
            paper = exact[0]
        else:
            non_supp = [
                p for p in pdfs
                if not any(k in p.name.lower() for k in ["supp", "supplement", "support", "supporting", "si", "补充"])
            ]
            if non_supp:
                paper = non_supp[0]
                if len(non_supp) > 1:
                    warnings.append("Multiple paper PDF candidates found; selected first by natural sort.")
            elif pdfs:
                paper = pdfs[0]
                warnings.append("Only supplementary-like PDFs found; selected first PDF as paper.")
            else:
                raise RuntimeError(f"No PDF files found in {pdf_dir}")

    if supp_pdf is not None:
        supp = Path(supp_pdf)
    else:
        supp = None
        if pdf_dir is not None:
            pdfs = sorted(
                [p for p in Path(pdf_dir).iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
                key=lambda p: natural_key(p.name),
            )
            exact_names = {
                "supplementary information.PDF",
                "supplementary.PDF",
                "supporting information.PDF",
                "si.PDF",
            }
            exact = [p for p in pdfs if p.name.lower() in exact_names]
            fuzzy = [
                p for p in pdfs
                if any(k in p.name.lower() for k in ["supp", "supplement", "support", "supporting", "si", "补充"])
            ]

            if exact:
                supp = exact[0]
            elif fuzzy:
                supp = fuzzy[0]
                if len(fuzzy) > 1:
                    warnings.append("Multiple supplementary PDF candidates found; selected first by natural sort.")
            else:
                warnings.append("No supplementary PDF found; only paper.PDF will be parsed.")

    check_pdf(paper, "paper_pdf")
    if supp is not None:
        check_pdf(supp, "supp_pdf")

    return paper, supp, warnings


def safe_extract_zip(zf: zipfile.ZipFile, out_dir: Path) -> None:
    """Extract zip safely, blocking Zip Slip style paths."""
    out_dir = Path(out_dir).resolve()
    for member in zf.infolist():
        target = (out_dir / member.filename).resolve()
        if not str(target).startswith(str(out_dir)):
            raise RuntimeError(f"Unsafe zip member path: {member.filename}")
    zf.extractall(out_dir)


class MinerUClient:
    def __init__(self, token: str, base_url: str = "https://mineru.net"):
        if not token:
            raise RuntimeError("Missing MinerU token. Fill MINERU_TOKEN or set environment variable MINERU_TOKEN.")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

    def create_upload_urls(
        self,
        files: List[Path],
        model_version: str = "vlm",
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "en",
    ) -> Tuple[str, List[str]]:
        url = f"{self.base_url}/api/v4/file-urls/batch"

        payload = {
            "files": [{"name": f.name, "data_id": f.stem} for f in files],
            "model_version": model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
        }

        r = requests.post(url, headers=self.headers, json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()

        if result.get("code") != 0:
            raise RuntimeError(f"MinerU create upload URLs failed: {result}")

        data = result.get("data", {}) or {}
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls") or []

        if not batch_id:
            raise RuntimeError(f"MinerU response missing batch_id: {result}")
        if not file_urls:
            raise RuntimeError(f"MinerU response missing file_urls: {result}")
        if len(file_urls) != len(files):
            raise RuntimeError(
                f"MinerU returned {len(file_urls)} upload URLs for {len(files)} files: {result}"
            )

        return str(batch_id), [str(u) for u in file_urls]

    def upload_files(self, files: List[Path], upload_urls: List[str]) -> None:
        for file_path, upload_url in zip(files, upload_urls):
            with file_path.open("rb") as f:
                r = requests.put(upload_url, data=f, timeout=600)
            r.raise_for_status()

    def poll_results(
        self,
        batch_id: str,
        timeout_seconds: int = 3600,
        interval_seconds: int = 10,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        start = time.time()
        last_result: Dict[str, Any] = {}

        terminal_states = {"done", "failed"}

        while True:
            r = requests.get(url, headers=self.headers, timeout=120)
            r.raise_for_status()
            last_result = r.json()

            if last_result.get("code") != 0:
                raise RuntimeError(f"MinerU poll failed: {last_result}")

            data = last_result.get("data", {}) or {}
            items = data.get("extract_result", []) or []

            if items:
                states = [str(item.get("state", "")).lower() for item in items]
                print("MinerU states:", states)
                if all(state in terminal_states for state in states):
                    return last_result

            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"MinerU polling timeout after {timeout_seconds} seconds. Last result: {last_result}")

            time.sleep(interval_seconds)

    def download_zip_and_find_md(self, full_zip_url: str, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)

        r = requests.get(full_zip_url, timeout=600)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content), "r") as zf:
            safe_extract_zip(zf, out_dir)

        full_md_candidates = sorted(out_dir.rglob("full.md"), key=lambda p: natural_key(str(p)))
        if full_md_candidates:
            return full_md_candidates[0]

        md_candidates = sorted(out_dir.rglob("*.md"), key=lambda p: natural_key(str(p)))
        if len(md_candidates) == 1:
            return md_candidates[0]

        if len(md_candidates) > 1:
            raise RuntimeError(f"Multiple markdown files found in zip; cannot decide: {[str(p) for p in md_candidates]}")

        raise RuntimeError(f"No markdown file found in MinerU zip extracted to: {out_dir}")


def save_clean_markdown_from_zip(
    client: MinerUClient,
    full_zip_url: str,
    role: str,
    target_path: Path,
) -> None:
    """
    下载 MinerU zip 到系统临时目录，提取 md，清洗后只写入目标 md。
    不会在 OUT_DIR 下保留 zip/images/full.md。
    """
    with tempfile.TemporaryDirectory() as tmp:
        role_dir = Path(tmp) / role
        role_dir.mkdir(parents=True, exist_ok=True)

        md_path = client.download_zip_and_find_md(full_zip_url, role_dir)
        text = md_path.read_text(encoding="utf-8", errors="ignore")

        if CLEAN_WEB_LIKE:
            text = clean_mineru_md_like_web(text, keep_image_lines=KEEP_IMAGE_LINES)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")


def run_step0(
    pdf_dir: Optional[Path],
    paper_pdf: Optional[Path],
    supp_pdf: Optional[Path],
    out_dir: Path,
    model_version: str,
    language: str,
    enable_formula: bool,
    enable_table: bool,
    timeout_seconds: int,
    poll_interval_seconds: int,
    overwrite: bool,
    token: Optional[str] = None,
    base_url: str = MINERU_BASE_URL,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "step0_manifest.json"
    paper_md_target = out_dir / "paper.md"
    supp_md_target = out_dir / "Supplementary information.md"

    paper, supp, warnings = pick_pdf_files(pdf_dir, paper_pdf, supp_pdf)

    expected_exists = paper_md_target.exists()
    if supp is not None:
        expected_exists = expected_exists and supp_md_target.exists()

    if expected_exists and not overwrite:
        manifest = {
            "ok": True,
            "stage": "skipped_exists",
            "message": "Markdown already exists. Set OVERWRITE=True to regenerate.",
            "input": {
                "pdf_dir": str(pdf_dir) if pdf_dir else None,
                "paper_pdf": str(paper),
                "supplementary_pdf": str(supp) if supp else None,
            },
            "output": {
                "paper_md": str(paper_md_target),
                "supplementary_md": str(supp_md_target) if supp_md_target.exists() else None,
                "manifest": str(manifest_path),
            },
            "warnings": warnings,
            "errors": [],
        }
        write_json(manifest_path, manifest)
        return manifest

    token = token or os.environ.get("MINERU_TOKEN")
    client = MinerUClient(token=token, base_url=base_url)

    files = [paper] + ([supp] if supp is not None else [])

    role_by_name = {normalize_name(paper.name): "paper"}
    if supp is not None:
        role_by_name[normalize_name(supp.name)] = "supplementary"

    errors: List[Dict[str, Any]] = []
    paper_state = "missing"
    supp_state = "missing"
    paper_zip_present = False
    supp_zip_present = False
    batch_id = ""

    try:
        batch_id, upload_urls = client.create_upload_urls(
            files=files,
            model_version=model_version,
            enable_formula=enable_formula,
            enable_table=enable_table,
            language=language,
        )

        print("MinerU batch_id:", batch_id)
        print("Uploading PDF files...")
        client.upload_files(files, upload_urls)

        print("Polling MinerU results...")
        result = client.poll_results(
            batch_id=batch_id,
            timeout_seconds=timeout_seconds,
            interval_seconds=poll_interval_seconds,
        )

        data = result.get("data", {}) or {}
        items = data.get("extract_result", []) or []

        if not items:
            raise RuntimeError(f"No extract_result returned from MinerU: {result}")

        for idx, item in enumerate(items):
            file_name = str(item.get("file_name") or "")
            state = str(item.get("state") or "").lower()
            err_msg = str(item.get("err_msg") or "")
            full_zip_url = str(item.get("full_zip_url") or "")

            role = role_by_name.get(normalize_name(file_name))

            # MinerU 返回 file_name 偶尔可能和本地文件名不完全一致。
            # 如果结果数量和上传文件数量一致，就按上传顺序兜底。
            if role is None:
                if len(items) == len(files):
                    role = "paper" if idx == 0 else "supplementary"
                    warnings.append(
                        f"MinerU result file_name '{file_name}' did not exactly match local name; "
                        f"used upload order fallback: idx={idx}, role={role}."
                    )
                else:
                    warnings.append(f"Unknown result file returned by MinerU: {file_name}")
                    continue

            if role == "paper":
                paper_state = state
                paper_zip_present = bool(full_zip_url)
            else:
                supp_state = state
                supp_zip_present = bool(full_zip_url)

            if state == "failed":
                errors.append({"file": file_name, "role": role, "state": state, "err_msg": err_msg})
                continue

            if state != "done":
                errors.append({"file": file_name, "role": role, "state": state, "err_msg": "Task did not finish with done state."})
                continue

            if not full_zip_url:
                errors.append({"file": file_name, "role": role, "state": state, "err_msg": "full_zip_url missing"})
                continue

            target = paper_md_target if role == "paper" else supp_md_target
            save_clean_markdown_from_zip(client, full_zip_url, role, target)

        ok = paper_md_target.exists() and not any(e.get("role") == "paper" for e in errors)
        if supp is not None:
            ok = ok and supp_md_target.exists() and not any(e.get("role") == "supplementary" for e in errors)

        manifest = {
            "ok": ok,
            "api": "MinerU",
            "api_version": "v4",
            "model_version": model_version,
            "clean_web_like": CLEAN_WEB_LIKE,
            "keep_image_lines": KEEP_IMAGE_LINES,
            "input": {
                "pdf_dir": str(pdf_dir) if pdf_dir else None,
                "paper_pdf": str(paper),
                "supplementary_pdf": str(supp) if supp else None,
            },
            "output": {
                "paper_md": str(paper_md_target) if paper_md_target.exists() else None,
                "supplementary_md": str(supp_md_target) if supp_md_target.exists() else None,
                "manifest": str(manifest_path),
            },
            "mineru": {
                "batch_id": batch_id,
                "paper_state": paper_state,
                "supplementary_state": supp_state,
                "paper_full_zip_url_present": paper_zip_present,
                "supplementary_full_zip_url_present": supp_zip_present,
            },
            "warnings": warnings,
            "errors": errors,
        }
        write_json(manifest_path, manifest)

        if not ok:
            raise RuntimeError(f"STEP0 failed. See manifest: {manifest_path}")

        return manifest

    except Exception as e:
        manifest = {
            "ok": False,
            "stage": "step0_pdf_to_md",
            "api": "MinerU",
            "api_version": "v4",
            "model_version": model_version,
            "clean_web_like": CLEAN_WEB_LIKE,
            "keep_image_lines": KEEP_IMAGE_LINES,
            "input": {
                "pdf_dir": str(pdf_dir) if pdf_dir else None,
                "paper_pdf": str(paper),
                "supplementary_pdf": str(supp) if supp else None,
            },
            "output": {
                "paper_md": str(paper_md_target) if paper_md_target.exists() else None,
                "supplementary_md": str(supp_md_target) if supp_md_target.exists() else None,
                "manifest": str(manifest_path),
            },
            "mineru": {
                "batch_id": batch_id,
                "paper_state": paper_state,
                "supplementary_state": supp_state,
                "paper_full_zip_url_present": paper_zip_present,
                "supplementary_full_zip_url_present": supp_zip_present,
            },
            "warnings": warnings,
            "errors": errors + [{"error": str(e)}],
        }
        write_json(manifest_path, manifest)
        raise


def run_step0_pdf_to_md(
    out_dir: Path,
    pdf_dir: Optional[Path] = None,
    paper_pdf: Optional[Path] = None,
    supp_pdf: Optional[Path] = None,
    model_version: str = "vlm",
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "en",
    timeout_seconds: int = 3600,
    poll_interval_seconds: int = 10,
    overwrite: bool = False,
    base_url: str = MINERU_BASE_URL,
) -> Dict[str, Any]:
    tk = os.environ.get("MINERU_TOKEN") or ENV_MINERU_TOKEN
    if not tk:
        tf = Path(__file__).resolve().parents[1] / ".mineru_token"
        if tf.exists():
            tk = tf.read_text(encoding="utf-8").strip()
    return run_step0(
        pdf_dir=pdf_dir,
        paper_pdf=paper_pdf,
        supp_pdf=supp_pdf,
        out_dir=out_dir,
        model_version=model_version,
        language=language,
        enable_formula=enable_formula,
        enable_table=enable_table,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        overwrite=overwrite,
        token=tk,
        base_url=base_url,
    )


if __name__ == "__main__":
    manifest = run_step0(
        pdf_dir=Path(PDF_DIR) if PDF_DIR else None,
        paper_pdf=Path(PAPER_PDF) if PAPER_PDF else None,
        supp_pdf=Path(SUPP_PDF) if SUPP_PDF else None,
        out_dir=Path(OUT_DIR),
        model_version=MODEL_VERSION,
        language=LANGUAGE,
        enable_formula=ENABLE_FORMULA,
        enable_table=ENABLE_TABLE,
        timeout_seconds=TIMEOUT_SECONDS,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        overwrite=OVERWRITE,
        token=MINERU_TOKEN or None,
    )

    print("STEP0 done.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
