from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .codex_client import call_codex_json
except Exception:  # pragma: no cover - keep final report usable without Codex
    call_codex_json = None


# =============================================================================
# Basic IO helpers
# =============================================================================

def read_json_if_exists(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fmt_list(value: Any, max_items: int = 20) -> str:
    items = _listify(value)
    if not items:
        return "未记录"
    texts: List[str] = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            name = item.get("name") or item.get("feature_id") or item.get("field") or item.get("name_cn")
            if name:
                texts.append(str(name))
            else:
                texts.append(json.dumps(item, ensure_ascii=False))
        else:
            texts.append(str(item))
    suffix = " ..." if len(items) > max_items else ""
    return ", ".join(texts) + suffix


def _fmt_metric(metrics: Dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "未生成"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _read_text_if_exists(path: Path, max_chars: Optional[int] = None) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n...(truncated)"
    return text


def _json_short(value: Any, max_chars: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = str(text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


# =============================================================================
# Final result summary
# =============================================================================

def collect_final_result(
    *,
    paper_id: str,
    workspace_root: Path,
    input_files: Dict[str, Any],
    paper_spec: Dict[str, Any],
    repro_spec: Dict[str, Any],
    reproduction_plan: Dict[str, Any],
    execution_manifest: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    paper_identity = paper_spec.get("paper_identity", {}) or {}
    plan_mode = reproduction_plan.get("reproduction_mode", {}) or {}
    understanding = reproduction_plan.get("paper_understanding", {}) or {}
    strategy = reproduction_plan.get("selected_reproduction_strategy", {}) or {}
    model_plan = reproduction_plan.get("model_execution_plan", {}) or {}
    evaluation_plan = reproduction_plan.get("evaluation_plan", {}) or {}
    readiness = paper_spec.get("readiness_audit", {}) or {}

    return {
        "paper_id": paper_id,
        "paper_name": paper_identity.get("paper_name") or reproduction_plan.get("paper_name") or paper_id,
        "workspace": str(workspace_root),
        "primary_report": "reports/paper_case_report.md",
        "input": input_files,
        "agent_status": execution_manifest.get("status", "unknown"),
        "selected_outputs_to_read": {
            "primary_case_report": "reports/paper_case_report.md",
            "machine_readable_result": "final/final_result.json",
            "execution_trace": "logs/agent_trace.jsonl",
        },
        "reproduction_mode": {
            "strict_paper_reproduction_possible": plan_mode.get("strict_paper_reproduction_possible"),
            "selected_mode": plan_mode.get("selected_mode"),
            "reason": plan_mode.get("reason"),
            "what_is_reproduced": plan_mode.get("what_is_reproduced", []),
            "what_is_not_reproduced": plan_mode.get("what_is_not_reproduced", []),
        },
        "paper_understanding": {
            "task": understanding.get("task", {}),
            "model": understanding.get("model", {}),
            "target_column": strategy.get("target_column"),
            "model_input_features": strategy.get("model_input_features", []),
            "metrics": evaluation_plan.get("metrics", []),
        },
        "readiness_audit": readiness,
        "execution": {
            "model_family": model_plan.get("model_family", ""),
            "model_name": model_plan.get("model_name", ""),
            "framework": model_plan.get("framework", ""),
            "metrics": metrics,
        },
        "artifacts": {
            "primary_report": "reports/paper_case_report.md",
            "code_dir": "code/",
            "data_dir": "data/",
            "model_dir": "model/",
            "results_dir": "results/",
            "reports_dir": "reports/",
            "final_dir": "final/",
            "logs_dir": "logs/",
            "agent_trace": "logs/agent_trace.jsonl",
            "metrics": "results/metrics.json",
            "predictions": "results/test_predictions.csv",
            "plot": "results/test_soh_true_vs_predicted.png",
        },
        "assumptions": _clean_assumptions(reproduction_plan.get("missing_details_and_filled_assumptions", []) or []),
        "warnings": reproduction_plan.get("warnings", []),
        "errors": [] if execution_manifest.get("status") == "success" else [execution_manifest.get("error", "")],
    }


# =============================================================================
# Assumption / risk extraction
# =============================================================================

def _is_empty_assumption_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return not str(item or "").strip()

    field = str(item.get("field") or item.get("assumption_item") or item.get("scope") or "").strip()
    value = item.get("filled_value", item.get("value", ""))
    why = str(item.get("why_needed") or item.get("reason") or "").strip()
    risk = str(item.get("risk") or item.get("impact") or "").strip()

    if not field and not why and not risk:
        return True
    if field.lower() in {"unknown", "none", "null"} and not why and not risk:
        return True
    if not str(value or "").strip() and not why and not risk:
        return True
    return False


def _clean_assumptions(raw_items: Iterable[Any]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in raw_items or []:
        if _is_empty_assumption_item(item):
            continue

        if isinstance(item, dict):
            cleaned.append({
                "field": item.get("field") or item.get("assumption_item") or item.get("scope") or "未命名假设",
                "filled_value": item.get("filled_value", item.get("value", "")),
                "why_needed": item.get("why_needed") or item.get("reason") or "用于保证复现流程可执行。",
                "risk": item.get("risk") or item.get("impact") or "可能导致结果与原论文严格数值复现存在差异。",
                "paper_reported": item.get("paper_reported", False),
            })
        else:
            cleaned.append({
                "field": "文本假设",
                "filled_value": str(item),
                "why_needed": "来自复现计划中的非结构化假设。",
                "risk": "需要人工复查其对严格复现的影响。",
                "paper_reported": False,
            })
    return cleaned


def _build_assumption_risk_items(
    final_result: Dict[str, Any],
    reproduction_plan: Dict[str, Any],
    model_alignment_text: str = "",
    training_report_text: str = "",
) -> List[Dict[str, Any]]:
    """Build high-quality assumption/risk items.

    This function intentionally does not rely only on
    missing_details_and_filled_assumptions, because generated plans may contain
    empty/unknown placeholders.
    """
    items: List[Dict[str, Any]] = []

    mode = reproduction_plan.get("reproduction_mode", {}) or {}
    dataset_plan = reproduction_plan.get("dataset_construction_plan", {}) or {}
    model_exec = reproduction_plan.get("model_execution_plan", {}) or {}
    model_def = model_exec.get("model_definition", {}) or {}
    loss_plan = model_exec.get("loss_plan", {}) or {}
    training_plan = model_exec.get("training_plan", {}) or {}

    # 1. Clean original assumptions from reproduction_plan.
    raw_assumptions = reproduction_plan.get("missing_details_and_filled_assumptions", []) or []
    for a in _clean_assumptions(raw_assumptions):
        items.append({
            "type": "filled_assumption",
            "item": a["field"],
            "implementation": a["filled_value"],
            "reason": a["why_needed"],
            "risk": a["risk"],
        })

    # 2. Strict reproduction boundary.
    if mode.get("strict_paper_reproduction_possible") is False:
        items.append({
            "type": "reproduction_boundary",
            "item": "strict_reproduction_not_available",
            "implementation": mode.get("selected_mode") or "executable_approximation",
            "reason": mode.get("reason") or "论文未提供严格复现所需的全部信息。",
            "risk": "当前结果不能声称为原论文数据集上的严格数值复现，只能说明论文方法结构和执行流程可运行。",
        })

    # 3. Synthetic dataset risk.
    dataset_mode = str(dataset_plan.get("dataset_mode") or "").lower()
    if "synthetic" in dataset_mode:
        items.append({
            "type": "dataset_approximation",
            "item": "feature_level_synthetic_dataset",
            "implementation": dataset_plan.get("dataset_mode"),
            "reason": "当前输入未包含论文原始电池循环数据，因此 Agent 构造特征级合成数据集以验证完整 pipeline。",
            "risk": "Test RMSE / Test R2 反映的是合成数据上的执行效果，不能与论文原始实验指标直接比较。",
        })

    # 4. Cell-level split assumption.
    cell_split = dataset_plan.get("cell_split", {}) or {}
    if cell_split:
        items.append({
            "type": "dataset_split_assumption",
            "item": "cell_level_train_val_test_split",
            "implementation": cell_split,
            "reason": "为了避免同一电池同时出现在训练集和测试集，Agent 使用 cell-level split。",
            "risk": "该划分可能与论文原始实验划分不同，因此数值结果不可直接对齐。",
        })

    # 5. Model approximation.
    model_text = json.dumps(model_def, ensure_ascii=False).lower()
    if (
        model_def.get("approximate_implementation_required") is True
        or "approximate" in model_text
        or "assumption" in model_text
    ):
        items.append({
            "type": "model_approximation",
            "item": "approximate_model_implementation",
            "implementation": model_def,
            "reason": model_def.get("reason") or "论文模型结构细节不足，Agent 生成可执行近似模型。",
            "risk": "生成模型的层数、隐藏维度或组件实现可能与论文原始实现存在差异。",
        })

    # 6. Physics / PDE / loss approximation.
    loss_text = json.dumps(loss_plan, ensure_ascii=False).lower()
    align_text = (model_alignment_text or "").lower()
    train_text = (training_report_text or "").lower()
    if any(k in loss_text + align_text + train_text for k in ["pde", "physics", "residual", "approx"]):
        items.append({
            "type": "loss_approximation",
            "item": "physics_or_pde_loss_approximation",
            "implementation": loss_plan or "见 model_alignment_report.md / training_report.md",
            "reason": "论文中的完整物理约束或退化方程实现细节可能未完全给出。",
            "risk": "物理约束项为可执行近似，不等同于论文完整 PDE / degradation equation 的严格实现。",
        })

    # 7. Training hyperparameter assumptions.
    training_text = json.dumps(training_plan, ensure_ascii=False).lower()
    if any(k in training_text for k in ["paper_reported\": false", "early_stopping", "seed", "epoch", "learning_rate"]):
        items.append({
            "type": "training_assumption",
            "item": "training_hyperparameters_or_schedule",
            "implementation": training_plan,
            "reason": "论文未完整提供当前可执行设置下的训练参数，Agent 填补默认训练配置。",
            "risk": "训练策略会影响最终指标，因此当前指标只能作为生成 pipeline 的执行结果。",
        })

    # Deduplicate by type + item.
    dedup: Dict[str, Dict[str, Any]] = {}
    for x in items:
        key = f"{x.get('type')}::{x.get('item')}"
        dedup[key] = x
    return list(dedup.values())


# =============================================================================
# Report context + Codex generation
# =============================================================================

def _build_report_context(
    *,
    final_result: Dict[str, Any],
    paper_spec: Dict[str, Any],
    repro_spec: Dict[str, Any],
    reproduction_plan: Dict[str, Any],
    workspace_root: Path,
) -> Dict[str, Any]:
    ws = Path(workspace_root)

    metrics = _safe_get(final_result, "execution", "metrics", default={}) or {}
    mode = final_result.get("reproduction_mode", {}) or {}
    dataset_plan = reproduction_plan.get("dataset_construction_plan", {}) or {}
    model_exec = reproduction_plan.get("model_execution_plan", {}) or {}
    strategy = reproduction_plan.get("selected_reproduction_strategy", {}) or {}

    model_alignment_text = _read_text_if_exists(ws / "reports" / "model_alignment_report.md", max_chars=4000)
    training_report_text = _read_text_if_exists(ws / "reports" / "training_report.md", max_chars=3000)

    assumption_risk_items = _build_assumption_risk_items(
        final_result=final_result,
        reproduction_plan=reproduction_plan,
        model_alignment_text=model_alignment_text,
        training_report_text=training_report_text,
    )

    return {
        "paper": {
            "paper_id": final_result.get("paper_id"),
            "paper_name": final_result.get("paper_name"),
            "agent_status": final_result.get("agent_status"),
        },
        "target": paper_spec.get("target_definition", {}) or {},
        "data_requirements": paper_spec.get("data_requirements", {}) or {},
        "feature_recipe_count": len(paper_spec.get("feature_recipe", []) or []),
        "model_from_paper": paper_spec.get("model_definition", {}) or {},
        "experiment_protocol": paper_spec.get("experiment_protocol", {}) or {},
        "readiness_audit": paper_spec.get("readiness_audit", {}) or final_result.get("readiness_audit", {}) or {},
        "reproduction_mode": mode,
        "dataset": {
            "dataset_mode": dataset_plan.get("dataset_mode"),
            "num_cells": dataset_plan.get("num_cells"),
            "cell_split": dataset_plan.get("cell_split"),
            "target_column": strategy.get("target_column") or model_exec.get("target_column"),
            "model_input_features": strategy.get("model_input_features") or model_exec.get("input_features"),
        },
        "model_execution": {
            "model_family": _safe_get(final_result, "execution", "model_family", default=""),
            "model_name": _safe_get(final_result, "execution", "model_name", default=""),
            "framework": _safe_get(final_result, "execution", "framework", default=""),
            "model_definition": model_exec.get("model_definition", {}),
            "loss_plan": model_exec.get("loss_plan", {}),
            "training_plan": model_exec.get("training_plan", {}),
        },
        "metrics": metrics,
        "assumption_risk_items": assumption_risk_items,
        "warnings": final_result.get("warnings", []),
        "errors": final_result.get("errors", []),
        "supporting_reports": {
            "model_alignment_report_excerpt": model_alignment_text,
            "training_report_excerpt": training_report_text,
        },
        "artifacts": final_result.get("artifacts", {}),
    }


def _validate_case_report_text(text: str) -> List[str]:
    required = [
        "## 1. 案例概览",
        "## 2. Agent 对论文的理解结果",
        "## 3. 复现可行性判断",
        "## 4. Agent 自动规划与执行过程",
        "## 5. 生成的复现代码",
        "## 6. 构造的数据集",
        "## 7. 模型运行结果",
        "## 8. 论文模型与生成模型的对齐情况",
        "## 9. 假设、近似与风险",
        "## 10. 训练报告摘要",
        "## 11. 可追溯性证据",
        "## 12. 案例结论",
    ]
    problems = [x for x in required if x not in text]
    if "unknown：填充值" in text or "**unknown**" in text:
        problems.append("报告中仍包含 unknown 空假设")
    return problems


def _generate_case_report_with_codex(report_context: Dict[str, Any]) -> str:
    if call_codex_json is None:
        raise RuntimeError("call_codex_json is unavailable")

    system_prompt = (
        "You are a rigorous Chinese technical report writer for a Data Agent competition. "
        "You write accurate, evidence-grounded Markdown reports. "
        "You must not fabricate paper results. "
        "You must clearly distinguish strict numerical reproduction from executable approximation."
    )

    user_prompt = f"""
请根据 report_context 生成一份中文 Markdown 案例报告。

硬性要求：
1. 只能使用 report_context 中的信息，不得编造论文结果。
2. 如果 strict_paper_reproduction_possible=false，必须明确说明这不是原论文数值复现。
3. 如果 dataset_mode 包含 synthetic，必须明确说明指标不能与论文原始数据集指标直接比较。
4. 必须重点写好：
   - 第 8 节：论文模型与生成模型的对齐情况
   - 第 9 节：假设、近似与风险
5. 不要输出空的 unknown 假设。
6. 输出必须是 JSON，格式如下：
{{
  "paper_case_report_md": "..."
}}

必须包含以下章节标题，标题文字必须完全一致：
## 1. 案例概览
## 2. Agent 对论文的理解结果
## 3. 复现可行性判断
## 4. Agent 自动规划与执行过程
## 5. 生成的复现代码
## 6. 构造的数据集
## 7. 模型运行结果
## 8. 论文模型与生成模型的对齐情况
## 9. 假设、近似与风险
## 10. 训练报告摘要
## 11. 可追溯性证据
## 12. 案例结论

report_context:
{json.dumps(report_context, ensure_ascii=False, indent=2)}
"""

    out = call_codex_json(system_prompt, user_prompt)
    text = out.get("paper_case_report_md", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Codex did not return paper_case_report_md")
    problems = _validate_case_report_text(text)
    if problems:
        raise RuntimeError("Codex report validation failed: " + " | ".join(problems))
    return text.strip() + "\n"


# =============================================================================
# Markdown generation helpers
# =============================================================================

def _alignment_table_from_plan(final_result: Dict[str, Any], reproduction_plan: Dict[str, Any]) -> List[str]:
    understanding = final_result.get("paper_understanding", {}) or {}
    model = understanding.get("model", {}) or {}
    model_exec = reproduction_plan.get("model_execution_plan", {}) or {}
    strategy = reproduction_plan.get("selected_reproduction_strategy", {}) or {}
    training = model_exec.get("training_plan", {}) or {}
    loss = model_exec.get("loss_plan", {}) or {}

    rows = [
        "| 项目 | 论文/抽取结果 | Agent 实现 | 一致性说明 |",
        "|---|---|---|---|",
        f"| 输入特征 | {_fmt_list(strategy.get('primary_features') or understanding.get('model_input_features'))} | {_fmt_list(model_exec.get('input_features') or understanding.get('model_input_features'))} | 以 Agent 抽取的 primary features 为默认输入 |",
        f"| 模型名称 | {_as_text(model.get('primary_model_name') or model.get('model_name') or final_result.get('execution', {}).get('model_name'))} | {_as_text(model_exec.get('model_name') or final_result.get('execution', {}).get('model_name'))} | 见 `reports/model_alignment_report.md` |",
        f"| 模型类型 | {_as_text(model.get('model_family') or final_result.get('execution', {}).get('model_family'))} | {_as_text(model_exec.get('model_family') or final_result.get('execution', {}).get('model_family'))} | 若论文细节不足，则为可执行近似 |",
        f"| 损失函数 | {_as_text(loss.get('total_loss') or loss.get('type') or '')} | {_as_text(loss.get('total_loss') or loss.get('type') or '见生成代码')} | 见训练代码与报告 |",
        f"| 训练参数 | {_as_text(training.get('hyperparameters') or training)} | {_as_text(training.get('hyperparameters') or training)} | 缺失值会记录为 filled assumptions |",
        f"| 评价指标 | {_fmt_list(final_result.get('paper_understanding', {}).get('metrics'))} | RMSE, R2, prediction plot | STEP4 输出统一校验 |",
    ]
    return rows


def _assumption_lines(assumptions: Iterable[Any]) -> List[str]:
    cleaned = _clean_assumptions(assumptions)
    if not cleaned:
        return ["- 未记录由 Agent 填补的关键假设。"]

    out: List[str] = []
    for item in cleaned:
        out.append(
            f"- **{item.get('field')}**：填充值 `{_json_short(item.get('filled_value'))}`；"
            f"原因：{item.get('why_needed') or '用于保证实验流程可执行。'}；"
            f"风险：{item.get('risk') or '可能导致结果与原论文严格数值复现存在差异。'}"
        )
    return out


def _assumption_risk_table(items: List[Dict[str, Any]]) -> List[str]:
    if not items:
        return [
            "未检测到明确的假设条目。但如果当前复现模式不是 strict reproduction，仍应以 `reproduction_mode.reason` 中的说明为准。",
        ]

    lines = [
        "| 类型 | 假设/近似项 | Agent 实现 | 原因 | 风险 |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            "| "
            + str(item.get("type", ""))
            + " | "
            + str(item.get("item", ""))
            + " | "
            + _json_short(item.get("implementation", ""), 220).replace("\n", " ")
            + " | "
            + str(item.get("reason", "")).replace("\n", " ")
            + " | "
            + str(item.get("risk", "")).replace("\n", " ")
            + " |"
        )
    return lines


# =============================================================================
# Report writers
# =============================================================================

def write_final_summary(path: Path, final_result: Dict[str, Any]) -> None:
    mode = final_result.get("reproduction_mode", {}) or {}
    understanding = final_result.get("paper_understanding", {}) or {}
    execution = final_result.get("execution", {}) or {}
    metrics = execution.get("metrics", {}) or {}

    lines = [
        "# Paper2SOH Agent Final Summary",
        "",
        "> 主要阅读入口：`reports/paper_case_report.md`。本文件是机器/人工均可快速查看的摘要。",
        "",
        "## 1. Input Paper",
        f"- Paper ID: {final_result.get('paper_id', '')}",
        f"- Paper Name: {final_result.get('paper_name', '')}",
        f"- Paper PDF: {final_result.get('input', {}).get('paper_pdf')}",
        f"- Supplementary PDF: {final_result.get('input', {}).get('supplementary_pdf')}",
        "",
        "## 2. Reproduction Mode",
        f"- Strict reproduction possible: {mode.get('strict_paper_reproduction_possible')}",
        f"- Selected mode: {mode.get('selected_mode')}",
        f"- Reason: {mode.get('reason')}",
        "",
        "## 3. Paper Understanding",
        f"- Task: {json.dumps(understanding.get('task', {}), ensure_ascii=False)}",
        f"- Model: {json.dumps(understanding.get('model', {}), ensure_ascii=False)}",
        f"- Target column: {understanding.get('target_column')}",
        f"- Input features: {understanding.get('model_input_features', [])}",
        "",
        "## 4. Execution Results",
        f"- Model family: {execution.get('model_family', '')}",
        f"- Model name: {execution.get('model_name', '')}",
        f"- Framework: {execution.get('framework', '')}",
        f"- Metrics: {json.dumps(metrics, ensure_ascii=False)}",
        "",
        "## 5. Generated Artifacts",
        "- `reports/paper_case_report.md`: primary case report for reviewers",
        "- `code/`: generated reproduction code",
        "- `data/`: generated dataset and train/val/test split",
        "- `model/`: trained model and scaler",
        "- `results/`: metrics, predictions, plot and history",
        "- `logs/`: agent trace and run manifest",
        "",
        "## 6. Assumptions and Limitations",
    ]
    lines.extend(_assumption_lines(final_result.get("assumptions", []) or []))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reproduction_story(path: Path, final_result: Dict[str, Any]) -> None:
    """Backward-compatible short story report. The primary report is paper_case_report.md."""
    mode = final_result.get("reproduction_mode", {}) or {}
    understanding = final_result.get("paper_understanding", {}) or {}
    execution = final_result.get("execution", {}) or {}
    artifacts = final_result.get("artifacts", {}) or {}

    lines = [
        "# Paper2SOH Reproduction Story",
        "",
        "> 本文件保留为兼容旧输出。每篇文章的主报告请阅读 `reports/paper_case_report.md`。",
        "",
        "## 1. What the Agent received",
        f"The Agent received paper `{final_result.get('paper_name', final_result.get('paper_id'))}` as PDF input.",
        "",
        "## 2. What the Agent understood",
        f"- Task: {json.dumps(understanding.get('task', {}), ensure_ascii=False)}",
        f"- Model: {json.dumps(understanding.get('model', {}), ensure_ascii=False)}",
        f"- Target column: {understanding.get('target_column')}",
        f"- Model input features: {understanding.get('model_input_features', [])}",
        "",
        "## 3. How the Agent planned reproduction",
        "The Agent decomposed the task into document parsing, paper understanding, reproduction specification, executable planning, code generation, model execution, and output validation.",
        "",
        "## 4. Reproduction mode",
        f"- Strict paper reproduction possible: {mode.get('strict_paper_reproduction_possible')}",
        f"- Selected mode: {mode.get('selected_mode')}",
        f"- Reason: {mode.get('reason')}",
        "",
        "## 5. Generated artifacts",
        f"- Primary report: {artifacts.get('primary_report')}",
        f"- Code: {artifacts.get('code_dir')}",
        f"- Dataset: {artifacts.get('data_dir')}",
        f"- Model: {artifacts.get('model_dir')}",
        f"- Results: {artifacts.get('results_dir')}",
        f"- Logs: {artifacts.get('logs_dir')}",
        "",
        "## 6. Execution results",
        f"- Framework: {execution.get('framework', '')}",
        f"- Metrics: {json.dumps(execution.get('metrics', {}), ensure_ascii=False)}",
        "",
        "## 7. Assumptions and limitations",
    ]
    lines.extend(_assumption_lines(final_result.get("assumptions", []) or []))
    lines += [
        "",
        "## 8. Conclusion",
        "This output package demonstrates an end-to-end Data Agent workflow from SOH paper PDF to auditable executable reproduction artifacts.",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_case_report(
    path: Path,
    *,
    final_result: Dict[str, Any],
    paper_spec: Dict[str, Any],
    repro_spec: Dict[str, Any],
    reproduction_plan: Dict[str, Any],
    workspace_root: Path,
) -> None:
    """Write the primary human-readable report for one paper.

    The function first builds a deterministic report_context.json. If
    CASE_REPORT_USE_CODEX is enabled, it asks Codex to write a polished report
    from that context. If Codex fails or the generated report is incomplete, it
    falls back to a deterministic template report.
    """
    ws = Path(workspace_root)
    path = Path(path)

    mode = final_result.get("reproduction_mode", {}) or {}
    understanding = final_result.get("paper_understanding", {}) or {}
    execution = final_result.get("execution", {}) or {}
    metrics = execution.get("metrics", {}) or {}
    readiness = paper_spec.get("readiness_audit", {}) or final_result.get("readiness_audit", {}) or {}

    target = paper_spec.get("target_definition", {}) or {}
    data_req = paper_spec.get("data_requirements", {}) or {}
    model_def = paper_spec.get("model_definition", {}) or {}
    protocol = paper_spec.get("experiment_protocol", {}) or {}
    feature_recipe = paper_spec.get("feature_recipe", []) or []
    strategy = reproduction_plan.get("selected_reproduction_strategy", {}) or {}
    dataset_plan = reproduction_plan.get("dataset_construction_plan", {}) or {}
    model_exec = reproduction_plan.get("model_execution_plan", {}) or {}
    warnings = final_result.get("warnings", []) or []
    errors = final_result.get("errors", []) or []

    model_alignment_text = _read_text_if_exists(ws / "reports" / "model_alignment_report.md", max_chars=3000)
    training_report_text = _read_text_if_exists(ws / "reports" / "training_report.md", max_chars=2200)

    report_context = _build_report_context(
        final_result=final_result,
        paper_spec=paper_spec,
        repro_spec=repro_spec,
        reproduction_plan=reproduction_plan,
        workspace_root=ws,
    )
    write_json(ws / "_work" / "report_context.json", report_context)

    use_codex = os.getenv("CASE_REPORT_USE_CODEX", "1").strip().lower() not in {"0", "false", "no"}
    generation_record = {
        "use_codex": use_codex,
        "method": "template_fallback",
        "error": "",
    }

    if use_codex:
        try:
            md = _generate_case_report_with_codex(report_context)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(md, encoding="utf-8")
            generation_record["method"] = "codex"
            write_json(ws / "_work" / "paper_case_report_generation.json", generation_record)
            return
        except Exception as e:
            generation_record["error"] = str(e)
            write_json(ws / "_work" / "paper_case_report_generation.json", generation_record)

    assumption_risk_items = report_context.get("assumption_risk_items", []) or []

    lines: List[str] = [
        f"# Paper Case Report: {final_result.get('paper_name') or final_result.get('paper_id')}",
        "",
        "> **这是本篇文章的主案例报告。** 评审或使用者优先阅读本文件；代码、数据、指标、日志均作为本报告的支撑材料。",
        "",
        "## 1. 案例概览",
        "",
        f"- 论文 ID：`{final_result.get('paper_id', '')}`",
        f"- 论文名称：{final_result.get('paper_name', '')}",
        f"- Agent 运行状态：`{final_result.get('agent_status', '')}`",
        f"- 复现模式：`{mode.get('selected_mode')}`",
        f"- 是否可严格复现：`{mode.get('strict_paper_reproduction_possible')}`",
        f"- 主报告文件：`reports/paper_case_report.md`",
        "",
        "**最终结论：** 本案例展示了 Agent 从 SOH 估计算法论文 PDF 出发，自动完成文档解析、论文理解、复现规划、代码生成、数据构造、模型执行和结果校验的全过程。若论文缺少原始数据或完整超参数，系统会明确标注为可执行近似复现，而不声称得到原论文数值复现。",
        "",
        "## 2. Agent 对论文的理解结果",
        "",
        "### 2.1 目标变量",
        "",
        f"- 目标名称：{target.get('target_name') or understanding.get('target_column') or '未记录'}",
        f"- 目标公式：{target.get('target_formula') or '未记录'}",
        f"- 标签来源：{target.get('target_source_signal') or target.get('label_source_signal') or '未记录'}",
        f"- 标签粒度：{target.get('label_granularity') or '未记录'}",
        f"- 抽取状态：{target.get('status') or '未记录'}",
        "",
        "### 2.2 数据与特征",
        "",
        f"- 论文/实验数据集：{_fmt_list(protocol.get('dataset_names'))}",
        f"- 需要的原始信号：{_fmt_list(data_req.get('required_signals'))}",
        f"- 需要的实验类型：{_fmt_list(data_req.get('required_experiment_types'))}",
        f"- Agent 抽取到的特征数量：{len(feature_recipe)}",
        f"- 默认模型输入特征：{_fmt_list(strategy.get('model_input_features') or understanding.get('model_input_features'))}",
        "",
        "### 2.3 模型结构",
        "",
        f"- 论文模型名称：{model_def.get('model_name') or _safe_get(understanding, 'model', 'primary_model_name', default='未记录')}",
        f"- 模型类型：{model_def.get('model_family') or execution.get('model_family') or '未记录'}",
        f"- 主要结构：{model_def.get('architecture') or _as_text(_safe_get(understanding, 'model', default={}))}",
        f"- 损失函数/损失项：{_fmt_list(model_def.get('loss_terms'))}",
        f"- 优化器/求解器：{model_def.get('optimizer_or_solver') or _safe_get(model_exec, 'training_plan', 'optimizer', default='未记录')}",
        "",
        "## 3. 复现可行性判断",
        "",
        f"- 严格复现是否可行：`{mode.get('strict_paper_reproduction_possible')}`",
        f"- Agent 选择的模式：`{mode.get('selected_mode')}`",
        f"- 选择原因：{mode.get('reason') or '未记录'}",
        f"- Readiness audit：`strict_reproduction_ready={readiness.get('strict_reproduction_ready')}`，`approximate_implementation_ready={readiness.get('approximate_implementation_ready')}`",
        "",
        "本系统将“严格复现”和“可执行近似复现”明确区分。若论文未提供原始数据、完整特征构造细节、完整网络结构或训练超参数，Agent 会把缺失项记录为 blocker/assumption，并生成可运行但带风险说明的复现包。",
        "",
        "## 4. Agent 自动规划与执行过程",
        "",
        "Agent 将任务拆解为以下步骤：",
        "",
        "1. **PDF 文档解析**：调用 MinerU 将主文和补充材料解析为 Markdown。",
        "2. **论文结构化理解**：抽取目标变量、特征工程、模型结构、训练协议、评价指标和公式证据。",
        "3. **复现规格生成**：判断严格复现是否可行，整理数据需求、模型需求和缺失项。",
        "4. **可执行复现计划生成**：将论文信息转化为可执行的 dataset/model/training/evaluation plan。",
        "5. **代码生成与执行**：生成数据集构造、模型定义、训练和评估代码，并运行完整 pipeline。",
        "6. **结果校验与报告生成**：校验输出文件、指标、预测结果和日志，生成本案例报告。",
        "",
        "## 5. 生成的复现代码",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `code/run_pipeline.py` | 一键运行生成的数据集、训练和评估流程 |",
        "| `code/dataset_generator.py` | 构造用于复现实验的数据集 |",
        "| `code/model_definitions.py` | 定义论文模型或可执行近似模型 |",
        "| `code/trainer.py` | 模型训练逻辑 |",
        "| `code/evaluator.py` | 指标计算、预测结果和图像输出 |",
        "| `code/README_RUN.md` | 生成代码的运行说明 |",
        "",
        "## 6. 构造的数据集",
        "",
        f"- 数据集模式：{dataset_plan.get('dataset_mode') or '未记录'}",
        f"- cell 数量：{dataset_plan.get('num_cells') or '未记录'}",
        f"- 划分策略：{json.dumps(dataset_plan.get('cell_split', {}), ensure_ascii=False)}",
        f"- 目标列：{strategy.get('target_column') or model_exec.get('target_column') or '未记录'}",
        f"- 输入特征：{_fmt_list(strategy.get('model_input_features') or model_exec.get('input_features'))}",
        "- 数据文件：`data/model_dataset.csv`、`data/train.csv`、`data/val.csv`、`data/test.csv`",
        "",
        "## 7. 模型运行结果",
        "",
        f"- 模型家族：{execution.get('model_family') or '未记录'}",
        f"- 模型名称：{execution.get('model_name') or '未记录'}",
        f"- 框架：{execution.get('framework') or '未记录'}",
        f"- Test RMSE：{_fmt_metric(metrics, 'test_RMSE')}",
        f"- Test R2：{_fmt_metric(metrics, 'test_R2')}",
        "- 指标文件：`results/metrics.json`",
        "- 预测结果：`results/test_predictions.csv`",
        "- 预测图：`results/test_soh_true_vs_predicted.png`",
        "",
        "## 8. 论文模型与生成模型的对齐情况",
        "",
        "本节用于说明 Agent 生成的模型与论文方法之间的对应关系。若论文缺少完整结构、原始代码或训练细节，Agent 会生成可执行近似实现，并在本节和第 9 节中说明复现边界。",
        "",
    ]
    lines.extend(_alignment_table_from_plan(final_result, reproduction_plan))

    if model_alignment_text:
        lines.extend([
            "",
            "### 8.1 模型对齐报告摘要",
            "",
            "```markdown",
            model_alignment_text,
            "```",
        ])

    lines.extend([
        "",
        "## 9. 假设、近似与风险",
        "",
        "下表列出 Agent 在生成可执行复现流程时使用的关键假设、近似实现和风险说明。这些内容用于明确复现边界，避免将可执行近似结果误表述为原论文严格数值复现。",
        "",
    ])
    lines.extend(_assumption_risk_table(assumption_risk_items))

    if warnings:
        lines.extend(["", "### 9.1 Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    if errors:
        lines.extend(["", "### 9.2 Errors", ""])
        for error in errors:
            if error:
                lines.append(f"- {error}")

    if training_report_text:
        lines.extend([
            "",
            "## 10. 训练报告摘要",
            "",
            "```markdown",
            training_report_text,
            "```",
        ])
    else:
        lines.extend([
            "",
            "## 10. 训练报告摘要",
            "",
            "未检测到 `reports/training_report.md`。请查看 `results/` 目录中的训练输出文件。",
        ])

    lines.extend([
        "",
        "## 11. 可追溯性证据",
        "",
        "| 证据文件 | 说明 |",
        "|---|---|",
        "| `final/final_result.json` | 机器可读的最终结果汇总 |",
        "| `logs/agent_trace.jsonl` | Agent 每一步执行轨迹、工具调用和错误记录 |",
        "| `results/metrics.json` | 模型测试指标 |",
        "| `results/test_predictions.csv` | 测试集真实值与预测值 |",
        "| `code/` | Agent 生成的完整复现代码 |",
        "| `_work/report_context.json` | 最终报告生成所用事实上下文 |",
        "| `_work/` | 内部中间产物，用于调试和复查 |",
        "",
        "## 12. 案例结论",
        "",
        "本案例的主要价值在于证明：Data Agent 不仅能够将复杂 SOH 估计算法论文 PDF 解析成结构化信息，还能进一步完成复现可行性判断、可执行计划生成、代码生成、数据构造、模型训练、指标输出和可追溯报告生成。该流程能够作为语料加工、科研论文结构化理解和算法复现自动化的综合案例。",
        "",
        "---",
        "",
        "**建议阅读顺序：** 先读本文件 `reports/paper_case_report.md`，再根据需要查看 `final/final_result.json`、`results/metrics.json`、`code/` 和 `logs/agent_trace.jsonl`。",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
