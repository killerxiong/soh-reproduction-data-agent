# SOH论文复现Data Agent

SOH论文复现Data Agent 是一个面向电池 SOH 估计算法论文的 Data Agent。输入一篇论文 PDF 后，系统会自动完成：

1. MinerU PDF 解析；
2. 论文结构化理解；
3. 复现规格生成；
4. 可执行复现计划生成；
5. 复现代码生成；
6. 数据集构造、模型训练与评估；
7. 单篇论文案例报告生成。

## 最重要的输出

每篇文章最终只需要优先阅读一个报告：

```text
outputs/<paper_id>/reports/paper_case_report.md
```

这个文件是**单篇论文的主案例报告**，用于比赛技术报告中的典型案例展示。它会整合：论文理解、复现模式、Agent 执行步骤、生成代码、构造数据集、模型结果、模型对齐、假设风险和可追溯证据。

其他文件都是支撑材料：

```text
outputs/<paper_id>/final/final_result.json       # 机器可读最终结果
outputs/<paper_id>/logs/agent_trace.jsonl        # Agent 执行日志
outputs/<paper_id>/results/metrics.json          # 指标
outputs/<paper_id>/code/                         # 生成代码
outputs/<paper_id>/data/                         # 构造数据集
```

## 输出目录结构

```text
outputs/
  paper1/
    input/                         # 输入 PDF 复制件
    code/                          # Agent 生成的复现代码
    data/                          # 构造数据集与 train/val/test
    model/                         # 训练模型和 scaler
    results/                       # metrics、预测结果、图、训练历史
    reports/
      paper_case_report.md          # 每篇文章的主案例报告，优先读这个
      reproduction_story.md         # 兼容旧版的简短故事报告
      model_alignment_report.md     # 生成模型与论文模型对齐说明
      training_report.md            # 训练报告
    final/
      final_result.json             # 机器可读最终结果
      final_summary.md              # 简要摘要
      reproduction_plan.json        # Agent 执行计划
    logs/
      agent_trace.jsonl             # 可追溯日志
      run_manifest.json             # STEP4 执行 manifest
    _work/                          # 中间产物，调试用
```

## PyCharm 单篇运行

打开：

```text
run_local_paper.py
```

修改顶部配置：

```python
PAPER_PDF = PROJECT_ROOT / "raw_data" / "paper1" / "PDF" / "1.pdf"
SUPP_PDF = None
OUT_ROOT = PROJECT_ROOT / "outputs"
PAPER_ID = "paper1"
RUN_STEPS = "all"
OVERWRITE = False
KEEP_WORK = True
```

然后右键运行 `run_local_paper.py`。

## 指定步骤运行

为了避免每次都从 MinerU 开始重跑，`RUN_STEPS` 支持指定步骤：

```python
RUN_STEPS = "step0"       # 只跑 PDF -> Markdown
RUN_STEPS = "step1"       # 只跑论文理解，复用 _work/paper.md
RUN_STEPS = "step2"       # 只跑复现规格，复用 paper_spec.json
RUN_STEPS = "step3"       # 只跑复现计划，复用 repro_spec.json
RUN_STEPS = "step4"       # 只跑代码生成和执行，复用 reproduction_plan.json
RUN_STEPS = "final"       # 只重新生成 final_result / paper_case_report
RUN_STEPS = "step4,final" # 跑 STEP4 后生成主案例报告
RUN_STEPS = "step1:step3" # 连续跑 STEP1 到 STEP3
```

注意：当 `RUN_STEPS` 不是 `all` 或从 `step0` 开始时，建议保持：

```python
OVERWRITE = False
```

否则已有 `_work/` 中间文件会被删除。

## 环境变量

建议在 PyCharm 的 `Run -> Edit Configurations -> Environment variables` 中配置：

```text
MINERU_TOKEN=你的 MinerU token
CODEX_API_KEY=你的 Codex/OpenAI-compatible key
CODEX_BASE_URL=https://sorryios.ai/codex
CODEX_MODEL_NAME=gpt-5.5
CODEX_PROXY=http://127.0.0.1:7897
```

如果不需要代理，不要设置 `CODEX_PROXY`。

也可以在 `run_local_paper.py` 顶部临时填写变量，但不要提交真实密钥到 GitHub。

## 命令行运行

```bash
python run_paper_agent.py \
  --paper_pdf raw_data/paper1/PDF/1.pdf \
  --supp_pdf raw_data/paper1/PDF/1-supp.pdf \
  --out_root outputs \
  --paper_id paper1 \
  --steps all
```

只重新生成主案例报告：

```bash
python run_paper_agent.py \
  --paper_pdf raw_data/paper1/PDF/1.pdf \
  --out_root outputs \
  --paper_id paper1 \
  --steps final
```

## 批量运行

```bash
python run_batch_agent.py --raw_data_dir raw_data --out_root outputs --steps all
```

批量运行会生成：

```text
outputs/batch_summary.json
outputs/submission_case_summary.md
```

`submission_case_summary.md` 是五篇论文案例的总览，每一行会链接/指向对应的 `reports/paper_case_report.md`。
