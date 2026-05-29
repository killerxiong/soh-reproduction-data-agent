from pathlib import Path


class Config:
    def __init__(self, out_dir, plan=None):
        self.plan = plan or {}
        self.out_dir = Path(out_dir)
        self.data_dir = self.out_dir / "data"
        self.model_dir = self.out_dir / "model"
        self.results_dir = self.out_dir / "results"

        self.paper_name = "Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis"
        self.paper_model_name = "Physics-informed neural network for battery SOH estimation"
        self.generated_model_class = "ExecutablePINN"
        self.model_family = "PINN"

        self.seed = 42
        self.num_cells = 10
        self.min_cycles = 180
        self.max_cycles = 300
        self.train_cells = [f"cell_{i:03d}" for i in range(1, 7)]
        self.val_cells = [f"cell_{i:03d}" for i in range(7, 9)]
        self.test_cells = [f"cell_{i:03d}" for i in range(9, 11)]

        self.input_features = [
            "current_mean",
            "current_standard_deviation",
            "current_kurtosis",
            "current_skewness",
            "current_window_charging_time",
            "current_window_accumulated_charge",
            "current_curve_slope",
            "current_curve_entropy",
            "voltage_mean",
            "voltage_standard_deviation",
            "voltage_kurtosis",
            "voltage_skewness",
            "voltage_window_charging_time",
            "voltage_window_accumulated_charge",
            "voltage_curve_slope",
            "voltage_curve_entropy",
            "cycle_index",
        ]
        self.secondary_features = [
            "chemistry_code",
            "nominal_capacity_ah",
            "upper_cutoff_voltage",
            "lower_cutoff_voltage",
            "ambient_temperature_c",
            "protocol_code",
            "cell_age_group",
        ]
        self.target_column = "soh"

        self.feature_range = (-1.0, 1.0)
        self.input_dim = len(self.input_features)
        self.batch_size = 256
        self.max_epochs = 300
        self.learning_rate = 1e-3
        self.weight_decay = 0.0
        self.early_stopping_patience = 40
        self.early_stopping_min_delta = 1e-6
        self.alpha_mono = 0.7
        self.beta_pde = 20.0
        self.grad_clip_norm = 5.0
        self.log_every = 10
        self.shuffle_train_batches = True

    @property
    def cycle_feature_index(self):
        return self.input_features.index("cycle_index")
