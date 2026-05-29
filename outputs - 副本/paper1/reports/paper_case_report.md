# Paper Case Report: Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis

> **这是本篇文章的主案例报告。** 评审或使用者优先阅读本文件；代码、数据、指标、日志均作为本报告的支撑材料。

## 1. 案例概览

- 论文 ID：`paper1`
- 论文名称：Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis
- Agent 运行状态：`success`
- 复现模式：`feature_level_synthetic_executable_reproduction`
- 是否可严格复现：`False`
- 主报告文件：`reports/paper_case_report.md`

**最终结论：** 本案例展示了 Agent 从 SOH 估计算法论文 PDF 出发，自动完成文档解析、论文理解、复现规划、代码生成、数据构造、模型执行和结果校验的全过程。若论文缺少原始数据或完整超参数，系统会明确标注为可执行近似复现，而不声称得到原论文数值复现。

## 2. Agent 对论文的理解结果

### 2.1 目标变量

- 目标名称：state-of-health (SOH)
- 目标公式：SOH = current available capacity / initial capacity
- 标签来源：available_capacity_per_cycle
- 标签粒度：per cycle
- 抽取状态：success

### 2.2 数据与特征

- 论文/实验数据集：XJTU battery dataset, TJU dataset, HUST dataset, MIT dataset
- 需要的原始信号：{"signal": "charging voltage time series near the end of charge", "details": "Voltage data from each cycle in the range [V_end - 0.2 V, V_end], where V_end is the charge cut-off voltage, are needed to extract statistical voltage-curve features.", "evidence": ["paper_md:0005"]}, {"signal": "charging current time series during constant-voltage charging near full charge", "details": "Current data from each cycle with current between 0.5 A and 0.1 A during constant-voltage charging are needed to extract statistical current-curve features.", "evidence": ["paper_md:0005"]}, {"signal": "cycle index or time", "details": "Cycle/time is used together with the extracted features as input to the PINN and to model degradation dynamics.", "evidence": ["paper_md:0003", "paper_md:0006"]}, {"signal": "SOH label per cycle", "details": "The supervised target is battery SOH for each cycle/sample. SOH is defined as the ratio of current available capacity to initial capacity.", "evidence": ["paper_md:0001", "paper_md:0003", "paper_md:0006"]}, {"signal": "initial capacity and current available capacity or equivalent capacity measurements", "details": "Required to compute SOH labels, since SOH is the ratio of current available capacity to initial capacity.", "evidence": ["paper_md:0001"]}
- 需要的实验类型：{"type": "cycle-level lithium-ion battery aging/degradation experiments", "details": "Data should consist of repeated charge/discharge cycles across battery life so that per-cycle degradation trajectory and SOH can be learned.", "evidence": ["paper_md:0004", "paper_md:0006"]}, {"type": "charging protocol that reaches full charge", "details": "The feature extraction method assumes the battery is fully charged so that the selected voltage and current ranges before full charge exist.", "evidence": ["paper_md:0002", "paper_md:0005"]}, {"type": "CC-CV or equivalent charging containing constant-current and constant-voltage portions", "details": "The method was designed around datasets that mostly contain CC-CV charging, and current features are selected from the constant-voltage charging stage.", "evidence": ["paper_md:0005"]}, {"type": "training/test data from multiple cells or batteries", "details": "The model is trained and evaluated by splitting batteries into training, validation, and test sets; small-sample variants can train on as few as one battery but still test on multiple batteries.", "evidence": ["paper_md:0006", "paper_md:0008"]}
- Agent 抽取到的特征数量：17
- 默认模型输入特征：current_mean, current_standard_deviation, current_kurtosis, current_skewness, current_window_charging_time, current_window_accumulated_charge, current_curve_slope, current_curve_entropy, voltage_mean, voltage_standard_deviation, voltage_kurtosis, voltage_skewness, voltage_window_charging_time, voltage_window_accumulated_charge, voltage_curve_slope, voltage_curve_entropy, cycle_index

### 2.3 模型结构

- 论文模型名称：Physics-informed neural network (PINN) for lithium-ion battery SOH estimation
- 模型类型：PINN
- 主要结构：The model takes extracted statistical features x and cycle/time t as inputs to estimate SOH. It consists of two neural-network-approximated components: a solution function f(·), implemented as neural network F(·), that maps cycle/features to SOH, and a nonlinear degradation-dynamics function g(·), implemented as neural network G(·), that models the SOH decay rate/degradation dynamics. The PINN is constrained by an empirical/state-space degradation equation expressed as a PDE, plus monotonic degradation behavior. During transfer learning, G(·) is frozen and only the solution component F(·) is fine-tuned.
- 损失函数/损失项：data loss, monotonicity loss, PDE/degradation-equation loss
- 优化器/求解器：{'name': 'Adam', 'paper_reported': True, 'learning_rate': 0.001, 'learning_rate_paper_reported': False, 'weight_decay': 0.0, 'weight_decay_paper_reported': False}

## 3. 复现可行性判断

- 严格复现是否可行：`False`
- Agent 选择的模式：`feature_level_synthetic_executable_reproduction`
- 选择原因：The original raw experimental datasets, exact train/validation/test battery identities for the 10-cell synthetic setting, exact learning rate, epochs, learning-rate schedule, early stopping, and complete PDE equation implementation details are not fully available from the supplied materials. Therefore the plan reproduces the reported input-output structure, engineered feature space, PINN family, and executable training/evaluation pipeline using explicit synthetic-data and training assumptions.
- Readiness audit：`strict_reproduction_ready=False`，`approximate_implementation_ready=True`

本系统将“严格复现”和“可执行近似复现”明确区分。若论文未提供原始数据、完整特征构造细节、完整网络结构或训练超参数，Agent 会把缺失项记录为 blocker/assumption，并生成可运行但带风险说明的复现包。

## 4. Agent 自动规划与执行过程

Agent 将任务拆解为以下步骤：

1. **PDF 文档解析**：调用 MinerU 将主文和补充材料解析为 Markdown。
2. **论文结构化理解**：抽取目标变量、特征工程、模型结构、训练协议、评价指标和公式证据。
3. **复现规格生成**：判断严格复现是否可行，整理数据需求、模型需求和缺失项。
4. **可执行复现计划生成**：将论文信息转化为可执行的 dataset/model/training/evaluation plan。
5. **代码生成与执行**：生成数据集构造、模型定义、训练和评估代码，并运行完整 pipeline。
6. **结果校验与报告生成**：校验输出文件、指标、预测结果和日志，生成本案例报告。

## 5. 生成的复现代码

| 文件 | 作用 |
|---|---|
| `code/run_pipeline.py` | 一键运行生成的数据集、训练和评估流程 |
| `code/dataset_generator.py` | 构造用于复现实验的数据集 |
| `code/model_definitions.py` | 定义论文模型或可执行近似模型 |
| `code/trainer.py` | 模型训练逻辑 |
| `code/evaluator.py` | 指标计算、预测结果和图像输出 |
| `code/README_RUN.md` | 生成代码的运行说明 |

## 6. 构造的数据集

- 数据集模式：feature_level_synthetic_dataset
- cell 数量：10
- 划分策略：{"split_unit": "cell", "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "train_cells_count": 6, "val_cells_count": 2, "test_cells_count": 2, "no_cell_overlap": true}
- 目标列：soh
- 输入特征：current_mean, current_standard_deviation, current_kurtosis, current_skewness, current_window_charging_time, current_window_accumulated_charge, current_curve_slope, current_curve_entropy, voltage_mean, voltage_standard_deviation, voltage_kurtosis, voltage_skewness, voltage_window_charging_time, voltage_window_accumulated_charge, voltage_curve_slope, voltage_curve_entropy, cycle_index
- 数据文件：`data/model_dataset.csv`、`data/train.csv`、`data/val.csv`、`data/test.csv`

## 7. 模型运行结果

- 模型家族：PINN
- 模型名称：executable_feature_level_PINN_F_G
- 框架：PyTorch
- Test RMSE：0.00777841
- Test R2：0.987349
- 指标文件：`results/metrics.json`
- 预测结果：`results/test_predictions.csv`
- 预测图：`results/test_soh_true_vs_predicted.png`

## 8. 论文模型与生成模型的对齐情况

| 项目 | 论文/抽取结果 | Agent 实现 | 一致性说明 |
|---|---|---|---|
| 输入特征 | current_mean, current_standard_deviation, current_kurtosis, current_skewness, current_window_charging_time, current_window_accumulated_charge, current_curve_slope, current_curve_entropy, voltage_mean, voltage_standard_deviation, voltage_kurtosis, voltage_skewness, voltage_window_charging_time, voltage_window_accumulated_charge, voltage_curve_slope, voltage_curve_entropy, cycle_index | current_mean, current_standard_deviation, current_kurtosis, current_skewness, current_window_charging_time, current_window_accumulated_charge, current_curve_slope, current_curve_entropy, voltage_mean, voltage_standard_deviation, voltage_kurtosis, voltage_skewness, voltage_window_charging_time, voltage_window_accumulated_charge, voltage_curve_slope, voltage_curve_entropy, cycle_index | 以 Agent 抽取的 primary features 为默认输入 |
| 模型名称 | Physics-informed neural network for battery SOH estimation | executable_feature_level_PINN_F_G | 见 `reports/model_alignment_report.md` |
| 模型类型 | PINN | PINN | 若论文细节不足，则为可执行近似 |
| 损失函数 |  | 见生成代码 | 见训练代码与报告 |
| 训练参数 | {"paper_reported": false, "optimizer": {"name": "Adam", "paper_reported": true, "learning_rate": 0.001, "learning_rate_paper_reported": false, "weight_decay": 0.0, "weight_decay_paper_reported": false}, "batch_size": {"value": 256, "paper_reported_context": "reported for XJTU; synthetic 10-cell reproduction adopts this value", "paper_reported_for_synthetic_setting": false}, "epochs": {"max_epochs": 300, "paper_reported": false}, "early_stopping": {"enabled": true, "monitor": "val_RMSE", "mode": "min", "patience": 40, "min_delta": 1e-06, "restore_best_weights": true, "paper_reported": false}, "seed": 42, "seed_paper_reported": false, "device": "cuda_if_available_else_cpu", "gradient_clipping": {"enabled": true, "max_norm": 5.0, "paper_reported": false}, "shuffle_train_batches": true, "save_best_model_to": "trained_model.pth", "log_every_n_epochs": 10, "required_training_outputs": ["trained_model.pth", "scaler.pkl", "training_history.csv", "training_report.md"]} | {"paper_reported": false, "optimizer": {"name": "Adam", "paper_reported": true, "learning_rate": 0.001, "learning_rate_paper_reported": false, "weight_decay": 0.0, "weight_decay_paper_reported": false}, "batch_size": {"value": 256, "paper_reported_context": "reported for XJTU; synthetic 10-cell reproduction adopts this value", "paper_reported_for_synthetic_setting": false}, "epochs": {"max_epochs": 300, "paper_reported": false}, "early_stopping": {"enabled": true, "monitor": "val_RMSE", "mode": "min", "patience": 40, "min_delta": 1e-06, "restore_best_weights": true, "paper_reported": false}, "seed": 42, "seed_paper_reported": false, "device": "cuda_if_available_else_cpu", "gradient_clipping": {"enabled": true, "max_norm": 5.0, "paper_reported": false}, "shuffle_train_batches": true, "save_best_model_to": "trained_model.pth", "log_every_n_epochs": 10, "required_training_outputs": ["trained_model.pth", "scaler.pkl", "training_history.csv", "training_report.md"]} | 缺失值会记录为 filled assumptions |
| 评价指标 | RMSE, R2 | RMSE, R2, prediction plot | STEP4 输出统一校验 |

### 8.1 模型对齐报告摘要

```markdown
# Model Alignment Report

1. Paper-reported model name: Physics-informed neural network for battery SOH estimation

2. Generated model class name: ExecutablePINN

3. Paper-reported input features: 16 engineered current/voltage charge-curve statistics plus cycle/time.

4. Generated model input features: current_mean, current_standard_deviation, current_kurtosis, current_skewness, current_window_charging_time, current_window_accumulated_charge, current_curve_slope, current_curve_entropy, voltage_mean, voltage_standard_deviation, voltage_kurtosis, voltage_skewness, voltage_window_charging_time, voltage_window_accumulated_charge, voltage_curve_slope, voltage_curve_entropy, cycle_index

5. Implemented architecture components:
   - Solution network F(.): 17 -> 60 -> 60 -> 32 -> 32 -> 1 with sine activations and sigmoid-scaled SOH output.
   - Dynamics network G(.): 4 -> 60 -> 60 -> 1 with sine activations.
   - Data MSE loss.
   - Monotonicity penalty using autograd derivative d(SOH)/d(cycle).
   - PDE/degradation residual loss matching d(SOH)/d(cycle) to G(.).

6. Approximated architecture components:
   - Exact degradation PDE is approximated by residual mean((d_soh_d_cycle - G(.))^2).
   - Raw signal feature extraction is replaced by feature-level synthetic generation.

7. Missing details from the paper:
   - Original raw datasets and exact 10-cell split.
   - Complete PDE implementation details.
   - Exact learning rate, epoch count, early stopping, and architecture context for this synthetic setting.

8. Fallback used: false

9. Reason for fallback, if any: none; a paper-specific executable PINN approximation was implemented.

10. Reproduction status: executable approximation, not strict numerical reproduction.

```

## 9. 假设、近似与风险

- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录
- **unknown**：填充值 `""`；原因：未记录；风险：未记录

## 10. 训练报告摘要

```markdown
# Training Report

Paper: Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis

Model family: PINN

Generated class: `ExecutablePINN`

Target: SOH

Input features: 17 engineered features/cycle variables.

Preprocessing: MinMaxScaler(feature_range=[-1, 1]) fit on train split only.

Optimizer: Adam(lr=0.001, weight_decay=0.0)

Loss: L = L_data + 0.7 * L_mono + 20.0 * L_pde.

Best validation RMSE: 0.004413 at epoch 284.

Executed epochs: 300.

Fallback used: false. This is an executable approximation of the paper-specific F/G PINN because exact raw data and full PDE equation details were unavailable.

```

## 11. 可追溯性证据

| 证据文件 | 说明 |
|---|---|
| `final/final_result.json` | 机器可读的最终结果汇总 |
| `logs/agent_trace.jsonl` | Agent 每一步执行轨迹、工具调用和错误记录 |
| `results/metrics.json` | 模型测试指标 |
| `results/test_predictions.csv` | 测试集真实值与预测值 |
| `code/` | Agent 生成的完整复现代码 |
| `_work/` | 内部中间产物，用于调试和复查 |

## 12. 案例结论

本案例的主要价值在于证明：Data Agent 不仅能够将复杂 SOH 估计算法论文 PDF 解析成结构化信息，还能进一步完成复现可行性判断、可执行计划生成、代码生成、数据构造、模型训练、指标输出和可追溯报告生成。该流程能够作为语料加工、科研论文结构化理解和算法复现自动化的综合案例。

---

**建议阅读顺序：** 先读本文件 `reports/paper_case_report.md`，再根据需要查看 `final/final_result.json`、`results/metrics.json`、`code/` 和 `logs/agent_trace.jsonl`。