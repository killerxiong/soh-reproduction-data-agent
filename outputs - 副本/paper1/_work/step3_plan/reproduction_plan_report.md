# STEP3 Executable Reproduction Plan Report

## Paper
- paper_id: paper1_wang_2024_pinn_battery_soh
- paper_name: Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis

## Selected reproduction mode
- strict paper reproduction possible: False
- selected mode: feature_level_synthetic_executable_reproduction
- what is reproduced: ['paper-reported model input-output structure', 'paper-reported engineered feature space', 'paper-reported primary model family', 'complete train/validation/test execution pipeline', 'cell-level train/validation/test split at 6/2/2 using exactly 10 cells', 'test RMSE, test R2, and SOH true-versus-predicted plot generation']
- what is not reproduced: ['paper original raw experimental dataset', 'paper exact numerical results', 'raw voltage/current/time feature extraction unless explicitly implemented', "paper's four-dataset 387-cell benchmark", "paper's exact 10-run averaged tables", "paper's exact source-domain/target-domain transfer-learning experiments"]

## Selected strategy
- primary features: ['current_mean', 'current_standard_deviation', 'current_kurtosis', 'current_skewness', 'current_window_charging_time', 'current_window_accumulated_charge', 'current_curve_slope', 'current_curve_entropy', 'voltage_mean', 'voltage_standard_deviation', 'voltage_kurtosis', 'voltage_skewness', 'voltage_window_charging_time', 'voltage_window_accumulated_charge', 'voltage_curve_slope', 'voltage_curve_entropy', 'cycle_index']
- secondary features: ['chemistry_code', 'nominal_capacity_ah', 'upper_cutoff_voltage', 'lower_cutoff_voltage', 'ambient_temperature_c', 'protocol_code', 'cell_age_group']
- model input features: ['current_mean', 'current_standard_deviation', 'current_kurtosis', 'current_skewness', 'current_window_charging_time', 'current_window_accumulated_charge', 'current_curve_slope', 'current_curve_entropy', 'voltage_mean', 'voltage_standard_deviation', 'voltage_kurtosis', 'voltage_skewness', 'voltage_window_charging_time', 'voltage_window_accumulated_charge', 'voltage_curve_slope', 'voltage_curve_entropy', 'cycle_index']
- target: soh
- reason: The paper reports 16 extracted current/voltage statistical features plus cycle/time as PINN inputs. Chemistry and protocol metadata are useful for synthetic variability but are not reported as direct model inputs, so they are generated as secondary features and excluded by default.

## Dataset construction plan
- num cells: 10
- split rule: {"split_unit": "cell", "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "train_cells_count": 6, "val_cells_count": 2, "test_cells_count": 2, "no_cell_overlap": true}
- cycle-SOH generation: {"min_cycles_per_cell": 180, "max_cycles_per_cell": 300, "initial_soh_range": [0.98, 1.02], "end_soh_range": [0.75, 0.9], "trajectory_shape": "nonlinear_degradation_with_optional_knee", "allow_small_regeneration_noise": true, "formula_template": "soh = initial_soh - a * normalized_cycle - b * max(0, normalized_cycle - knee)^2 + noise", "constraints": ["SOH should generally decrease with cycle_index", "SOH must be clipped to [0.70, 1.05]", "different cells must not have identical degradation trajectories", "cycle_index must be integer starting at 1 for each cell"]}
- feature generation method: feature_target_correlated_synthetic_generation
- labels: {"target_column": "soh", "required_label_columns": ["soh", "capacity_loss"], "optional_label_columns": ["Q_cell_loss"], "formulas": {"capacity_loss": "1 - soh", "Q_cell_loss": "capacity_loss", "soh": "1 - capacity_loss"}}
- output files: ['cell_metadata.csv', 'cycle_soh_trajectories.csv', 'features.csv', 'labels.csv', 'model_dataset.csv', 'train.csv', 'val.csv', 'test.csv']

## Model execution plan
- model family: PINN
- model implementation: {"paper_reported": false, "executable_spec": {"class_name": "ExecutablePINN", "input_dim": 17, "output_dim": 1, "solution_network_F": {"description": "Approximates f(t_i, x_i) mapping cycle/features to SOH.", "layers": [{"type": "LinearSin", "in_features": 17, "out_features": 60, "activation": "sin"}, {"type": "LinearSin", "in_features": 60, "out_features": 60, "activation": "sin"}, {"type": "Linear", "in_features": 60, "out_features": 32, "activation": "identity"}, {"type": "LinearSin", "in_features": 32, "out_features": 32, "activation": "sin"}, {"type": "Linear", "in_features": 32, "out_features": 1, "activation": "sigmoid_scaled_to_[0,1.2]"}]}, "dynamics_network_G": {"description": "Approximates g(.) degradation dynamics term from SOH, cycle, and selected latent derivatives.", "input_construction": "concat(predicted_soh, scaled_cycle_index, first_order_feature_summary, capacity_loss_proxy)", "input_dim": 4, "layers": [{"type": "LinearSin", "in_features": 4, "out_features": 60, "activation": "sin"}, {"type": "LinearSin", "in_features": 60, "out_features": 60, "activation": "sin"}, {"type": "Linear", "in_features": 60, "out_features": 1, "activation": "identity"}]}, "initialization": "torch_default_initialization_with_seed_42", "forward_outputs": ["predicted_soh", "dynamics_rate"], "inference_output": "predicted_soh"}, "paper_basis": ["The paper reports F(.) and G(.) as small fully connected neural networks.", "Supplementary snippets show Linear+Sin and Linear layers for PINN/MLP structures, but not enough complete context to guarantee exact implementation in this 10-cell synthetic setting."]}
- input features: ['current_mean', 'current_standard_deviation', 'current_kurtosis', 'current_skewness', 'current_window_charging_time', 'current_window_accumulated_charge', 'current_curve_slope', 'current_curve_entropy', 'voltage_mean', 'voltage_standard_deviation', 'voltage_kurtosis', 'voltage_skewness', 'voltage_window_charging_time', 'voltage_window_accumulated_charge', 'voltage_curve_slope', 'voltage_curve_entropy', 'cycle_index']
- target: soh
- preprocessing: {"feature_columns": ["current_mean", "current_standard_deviation", "current_kurtosis", "current_skewness", "current_window_charging_time", "current_window_accumulated_charge", "current_curve_slope", "current_curve_entropy", "voltage_mean", "voltage_standard_deviation", "voltage_kurtosis", "voltage_skewness", "voltage_window_charging_time", "voltage_window_accumulated_charge", "voltage_curve_slope", "voltage_curve_entropy", "cycle_index"], "target_column": "soh", "scaler": "StandardScaler", "fit_scaler_on": "train_only", "transform": ["train", "val", "test"], "handle_missing_values": "raise_error_unless_user_allows_imputation"}
- loss: {"paper_reported_terms": ["L_data", "L_mono", "L_PDE"], "executable_loss": {"total_loss_formula": "L = L_data + alpha * L_mono + beta * L_pde", "L_data": {"type": "MSELoss", "formula": "mean((soh_true - soh_pred)^2)"}, "L_mono": {"type": "monotonicity_penalty", "formula": "mean(relu(d_soh_pred_d_cycle)^2)", "implementation": "Use torch.autograd.grad on scaled cycle_index input. Penalize positive derivative because degradation SOH should not increase with cycle."}, "L_pde": {"type": "degradation_dynamics_residual", "formula": "mean((d_soh_pred_d_cycle - G_input_rate)^2)", "implementation": "Compute d_soh_pred_d_cycle with autograd; pass constructed dynamics inputs through G; penalize mismatch."}, "alpha": 0.7, "beta": 20.0, "loss_weights_paper_reported_for_synthetic_setting": false, "reduction": "mean"}}
- training plan: {"paper_reported": false, "optimizer": {"name": "Adam", "paper_reported": true, "learning_rate": 0.001, "learning_rate_paper_reported": false, "weight_decay": 0.0, "weight_decay_paper_reported": false}, "batch_size": {"value": 256, "paper_reported_context": "reported for XJTU; synthetic 10-cell reproduction adopts this value", "paper_reported_for_synthetic_setting": false}, "epochs": {"max_epochs": 300, "paper_reported": false}, "early_stopping": {"enabled": true, "monitor": "val_RMSE", "mode": "min", "patience": 40, "min_delta": 1e-06, "restore_best_weights": true, "paper_reported": false}, "seed": 42, "seed_paper_reported": false, "device": "cuda_if_available_else_cpu", "gradient_clipping": {"enabled": true, "max_norm": 5.0, "paper_reported": false}, "shuffle_train_batches": true, "save_best_model_to": "trained_model.pth", "log_every_n_epochs": 10, "required_training_outputs": ["trained_model.pth", "scaler.pkl", "training_history.csv", "training_report.md"]}
- validation plan: {"validation_split": "cell-level validation set containing cell_007 and cell_008", "validation_frequency": "each_epoch", "metrics": ["val_RMSE", "val_R2", "val_MAE"], "model_selection": "select checkpoint with lowest val_RMSE", "no_test_leakage": true, "validation_outputs": ["validation_predictions.csv", "training_history.csv"]}
- testing plan: {"test_split": "cell-level test set containing cell_009 and cell_010", "load_checkpoint": "trained_model.pth", "apply_train_fitted_scaler": true, "predict_columns": ["predicted_soh"], "required_prediction_columns": ["cell_id", "cycle_index", "true_soh", "predicted_soh"], "metrics": ["test_RMSE", "test_R2", "test_MAE", "test_MAPE"], "plots": ["test_soh_true_vs_predicted.png"], "save_outputs": ["metrics.json", "test_predictions.csv", "test_soh_true_vs_predicted.png"]}

## Evaluation plan
- test RMSE
- test R2
- true-vs-predicted SOH plot
- details: {"metrics": [{"name": "RMSE", "split": "test", "formula": "sqrt(mean((y_true - y_pred)^2))", "output_key": "test_RMSE"}, {"name": "R2", "split": "test", "formula": "1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)", "output_key": "test_R2"}], "soh_conversion_rule": {"if_target_is_soh": "use prediction directly", "if_target_is_capacity_loss": "predicted_soh = 1 - predicted_capacity_loss", "if_target_is_Q_cell_loss": "predicted_soh = 1 - predicted_Q_cell_loss"}, "plots": [{"name": "test_soh_true_vs_predicted", "type": "scatter_or_line", "x": "cycle_index or sample_index", "y_true": "true_soh", "y_pred": "predicted_soh", "split": "test", "output_file": "test_soh_true_vs_predicted.png"}], "prediction_outputs": ["test_predictions.csv"]}

## Filled assumptions
| Field | Filled value | Why needed | Risk |
|---|---|---|---|
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |
|  | "" |  |  |

## Evidence from paper/spec
- evidence count: 13