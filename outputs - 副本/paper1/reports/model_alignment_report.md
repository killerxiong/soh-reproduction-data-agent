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
