
# Model Hyperparameters
# Stage 1: Standard Qlib LGBModel (Experiment 8 Optimized)
# UPDATED 2026-01-15: Reduced regularization/constraints for robustness on small data
MODEL_PARAMS_STAGE1 = {
    "loss": "mse",
    "colsample_bytree": 0.6116,
    "learning_rate": 0.02,
    "subsample": 0.6,
    "lambda_l1": 0.1,  # Reduced from 0.5
    "lambda_l2": 0.1,  # Reduced from 0.5
    "max_depth": 5,
    "num_leaves": 19,
    "min_data_in_leaf": 20, # Reduced from 34
    "early_stopping_rounds": 100,
    "num_boost_round": 1000
    # Seed injected at runtime
}

# Stage 2: Refined Native LightGBM (Selected Features)
MODEL_PARAMS_STAGE2 = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.03, # Slower learning for robustness
    "num_leaves": 15,      # Smaller trees
    "colsample_bytree": 0.6,
    "subsample": 0.6,
    "reg_alpha": 1.0,      # Stronger L1
    "reg_lambda": 1.0,     # Stronger L2
    "n_jobs": -1,
    "verbosity": -1,
    # Seed injected at runtime
}
