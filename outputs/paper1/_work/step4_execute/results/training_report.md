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
