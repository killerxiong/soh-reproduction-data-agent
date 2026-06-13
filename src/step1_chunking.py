import re
from pathlib import Path
from typing import Dict, List, Any

URL_RE = re.compile(r"https?://[^\s\]\)>,]+", re.IGNORECASE)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_text(errors="ignore")


def extract_urls(text: str) -> List[str]:
    return sorted(set(u.strip(".,;)]>") for u in URL_RE.findall(text or "")))


def split_markdown_sections(text: str) -> List[Dict[str, str]]:
    lines = (text or "").splitlines()
    sections = []
    cur_title = "__preamble__"
    cur = []
    for ln in lines:
        if ln.strip().startswith("#"):
            if cur:
                sections.append({"section_title": cur_title, "text": "\n".join(cur).strip()})
            cur_title = ln.strip().lstrip("#").strip() or "untitled"
            cur = []
        else:
            cur.append(ln)
    if cur:
        sections.append({"section_title": cur_title, "text": "\n".join(cur).strip()})
    return [s for s in sections if s["text"]]


def build_paper_chunks(inputs: Dict[str, Any], max_chars: int = 6000) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for source_key, source_name in (("paper_text", "paper_md"), ("supp_text", "supplementary_md")):
        sections = split_markdown_sections(inputs.get(source_key, ""))
        idx = 1
        for sec in sections:
            text = sec["text"]
            start = 0
            while start < len(text):
                part = text[start:start + max_chars]
                chunk_id = f"{source_name}:{idx:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "section_title": sec["section_title"],
                    "text": part,
                })
                idx += 1
                start += max_chars
    return chunks


def load_paper_inputs(paper_dir: Path) -> Dict[str, Any]:
    paper_md = paper_dir / "paper.md"
    supp_md = paper_dir / "Supplementary information.md"
    paper_text = _read_text(paper_md)
    supp_text = _read_text(supp_md)
    merged = (paper_text or "") + "\n\n" + (supp_text or "")
    return {
        "paper_dir": str(paper_dir),
        "paper_md": str(paper_md),
        "supplementary_md": str(supp_md),
        "paper_exists": paper_md.exists(),
        "supplementary_exists": supp_md.exists(),
        "paper_text": paper_text,
        "supp_text": supp_text,
        "urls": extract_urls(merged),
    }
