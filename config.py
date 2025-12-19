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
    # Device selection (default CPU; set to "cuda:0" to use GPU)
    "device": "cpu",
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


EXPERIMENTS = [
    {
        "name": "baseline",
        "mode": CompileMode.NONE,
        "compile_training_backbone": False,
        "trigger_every": 0,
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": False,
        "infer_output_index": -1,
    },
    {
        "name": "async_compile",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": True,
        "trigger_every": 1,
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": True,
        "infer_output_index": -1,
    },
    {
        "name": "async_quant",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": True,
        "trigger_every": 1,
        "enable_diff_check": True,
        "compressors": ["quant"],
        "async_warmup": True,
        "infer_output_index": -1,
    },
    {
        "name": "async_quant_weight_only",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": True,
        "trigger_every": 1,
        "enable_diff_check": True,
        "compressors": ["quant"],
        "async_warmup": True,
        "infer_output_index": -1,
        "quant_mode": "weight_only",
    },
    # {
    #     "name": "async_quant_tensorrt_int8",
    #     "mode": CompileMode.ASYNC,
    #     "compile_training_backbone": True,
    #     "trigger_every": 1,
    #     "enable_diff_check": True,
    #     "compressors": ["quant"],
    #     "async_warmup": True,
    #     "infer_output_index": -1,
    #     "quant_mode": "tensorrt_int8",
    # },
]
