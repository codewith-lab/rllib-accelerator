"""
Centralized hyper-parameter definitions for RLlib accelerator experiments.
"""

from framework.policy_manager import CompileMode


DEFAULT_HPARAMS = {
    # Environment & training loop
    "env_id": "CartPole-v1",
    "num_epochs": 200,
    "train_batch_size": 2000,
    "lr": 1e-5,
    "seed": 42,
    "num_rollout_workers": 4,
    "rollout_fragment_length": 500,
    # Model architecture
    "hidden_dim": 2048,
    "hidden_depth": 8,
    "use_residual": True,
    # Compile / Quant settings
    "compile_backend": "inductor",
    "compile_diff_threshold": 1e-4,
    "quant_diff_threshold": 5e-4,
    "min_epoch_before_compress": 30,   # rule out early‑training instability
    "quant_mode": "dynamic",  
    "quant_trt_calib_batches": 4,
    "quant_trt_calib_batch_size": 64,
    # Learning rate decay (optional)
    "lr_decay": {
        "enabled": True,
        "gamma": 0.9,           # Decay factor
        "step_epochs": 100,     # Decay interval in epochs
        "min_lr": 1e-6,         # Minimum learning rate
    },
    # Logging
    "use_wandb": True,
    "wandb_project": "rllib-accelerator",
    "wandb_group": "compile_quant_baseline_inference_speed_comparison_layer=8_dim=2048",
}


_COMMON_EXPERIMENT = {
    "infer_output_index": -1,
    "device": "cpu",
}

_BASELINE_EXPERIMENT = {
    **_COMMON_EXPERIMENT,
    "mode": CompileMode.NONE,
    "compile_training_backbone": False,
    "trigger_every": 0,
    "async_warmup": False,
}

_ASYNC_EXPERIMENT = {
    **_COMMON_EXPERIMENT,
    "mode": CompileMode.ASYNC,
    "compile_training_backbone": True,
    "trigger_every": 1,
    "async_warmup": True,
}

EXPERIMENTS = [
    {
        **_BASELINE_EXPERIMENT,
        "name": "baseline",
        "enable_diff_check": False,
        "compressors": ["compile"],
    },
    {
        **_ASYNC_EXPERIMENT,
        "name": "async_compile",
        "enable_diff_check": False,
        "compressors": ["compile"],
    },
    {
        **_ASYNC_EXPERIMENT,
        "name": "async_quant",
        "enable_diff_check": True,
        "compressors": ["quant"],
    },
    {
        **_ASYNC_EXPERIMENT,
        "name": "async_quant_weight_only",
        "enable_diff_check": True,
        "compressors": ["quant"],
        "quant_mode": "weight_only",
        "device": "cuda:0",
    },
    # {
    #     **_ASYNC_EXPERIMENT,
    #     "name": "async_quant_tensorrt_int8",
    #     "enable_diff_check": True,
    #     "compressors": ["quant"],
    #     "quant_mode": "tensorrt_int8",
    #     "device": "cuda:0",
    # },
]
