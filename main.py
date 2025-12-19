import os
import random

import numpy as np
import ray
import torch
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig

from framework.trainer import Trainer
from compression.compile_compressor import CompileCompressor
from compression.quant_compressor import QuantCompressor
from config import DEFAULT_HPARAMS, EXPERIMENTS

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Custom ModelV2.*")
warnings.filterwarnings("ignore", message=".*Install gputil.*")
warnings.filterwarnings("ignore", message=".*remote_workers.*")

# Register the global model with RLlib
from models.policy import CustomPolicyNet    # noqa


def resolve_device(config_device: str):
    requested = os.environ.get("ACCEL_DEVICE", config_device)
    normalized = requested.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        print(f"[main] ⚠️ Requested device '{requested}' unavailable, fallback to CPU.")
        return "cpu"
    return normalized if normalized.startswith("cuda") or normalized == "cpu" else requested


def apply_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_schedule(hparams):
    decay_conf = hparams.get("lr_decay") or {}
    if not decay_conf.get("enabled"):
        return None
    base_lr = hparams["lr"]
    gamma = float(decay_conf.get("gamma", 0.5))
    step_epochs = max(1, int(decay_conf.get("step_epochs", 1)))
    min_lr = float(decay_conf.get("min_lr", 0.0))
    total_epochs = hparams["num_epochs"]
    steps_per_epoch = max(1, hparams["train_batch_size"])

    schedule = [[0, base_lr]]
    current_lr = base_lr
    epoch = step_epochs
    while epoch <= total_epochs:
        current_lr = max(min_lr, current_lr * gamma)
        schedule.append([epoch * steps_per_epoch, current_lr])
        epoch += step_epochs

    return schedule if len(schedule) > 1 else None


def build_config(hidden_layers, device: str, hparams):
    use_gpu = device.startswith("cuda") and torch.cuda.is_available()
    lr_schedule = build_lr_schedule(hparams)
    training_kwargs = {
        "model": {
            "custom_model": "custom_policy",
            "fcnet_hiddens": hidden_layers,
            "custom_model_config": {
                "use_residual": hparams["use_residual"],
                "device": device,
            },
        },
        "train_batch_size": hparams["train_batch_size"],
        "lr": hparams["lr"],
    }
    if lr_schedule is not None:
        training_kwargs["lr_schedule"] = lr_schedule
    config = (
        PPOConfig()
        .environment(hparams["env_id"])
        .framework("torch")
        .resources(
            num_gpus=1 if use_gpu else 0,
            num_gpus_per_worker=0.25 if use_gpu else 0,
        )
        .training(**training_kwargs)
    )
    
    # Disable new API stack for backward compatibility with custom_model
    try:
        config = config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    except AttributeError:
        pass  # Older Ray versions don't provide this
    
    # Handle Ray API differences across versions
    try:
        # Ray >= 2.10 uses env_runners
        config = config.env_runners(
            num_env_runners=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    except AttributeError:
        # Ray < 2.10 falls back to rollouts
        config = config.rollouts(
            num_rollout_workers=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    seed = hparams.get("seed")
    if seed is not None:
        config.seed = seed
    return config


def build_compressors(exp_conf, device, hparams):
    names = exp_conf.get("compressors", ["compile"])
    comps = []
    for name in names:
        if name == "compile":
            comps.append(
                CompileCompressor(
                    backend=hparams["compile_backend"],
                    diff_threshold=hparams["compile_diff_threshold"],
                    device=device,
                    recompile_every=hparams.get("compile_recompile_every", 2),
                    sparsity_change_threshold=hparams.get("compile_sparsity_change_threshold", 0.05),
                )
            )
        elif name == "quant":
            quant_mode = exp_conf.get("quant_mode", hparams.get("quant_mode", "dynamic"))
            comps.append(
                QuantCompressor(
                    diff_threshold=hparams["quant_diff_threshold"],
                    mode=quant_mode,
                    trt_calib_batches=hparams.get("quant_trt_calib_batches", 4),
                    trt_calib_batch_size=hparams.get("quant_trt_calib_batch_size", 64),
                    device=device,
                )
            )
        elif name == "prune":
            # Mask-Based (Unstructured) Pruning
            from compression.mask_prune_compressor import MaskPruneCompressor
            comps.append(
                MaskPruneCompressor(
                    prune_ratio=hparams.get("prune_ratio", 0.25),
                    diff_threshold=hparams.get("prune_diff_threshold", 1e-3),
                    technique=hparams.get("prune_technique", "magnitude"),
                    schedule=hparams.get("prune_schedule", "iterative"),
                    prune_steps=hparams.get("prune_steps", 10),
                )
            )
        elif name == "prune+compile":
            # Mask-Based Pruning + Compile
            from compression.mask_prune_compressor import MaskPruneCompressor
            comps.append(
                MaskPruneCompressor(
                    prune_ratio=hparams.get("prune_ratio", 0.25),
                    diff_threshold=hparams.get("prune_diff_threshold", 1e-3),
                    technique=hparams.get("prune_technique", "magnitude"),
                    schedule=hparams.get("prune_schedule", "iterative"),
                    prune_steps=hparams.get("prune_steps", 10),
                )
            )
            comps.append(
                CompileCompressor(
                    backend=hparams["compile_backend"],
                    diff_threshold=hparams["compile_diff_threshold"],
                    device=device,
                    recompile_every=hparams.get("compile_recompile_every", 2),
                    sparsity_change_threshold=hparams.get("compile_sparsity_change_threshold", 0.05),
                )
            )
        else:
            raise ValueError(f"Unknown compressor name: {name}")
    return comps


def _collect_trt_calibration(algo, target_samples: int, batch_size: int):
    """Collect on-policy tensors exactly as fed into the model for TensorRT calibration."""
    worker = algo.workers.local_worker()
    policy = worker.get_policy()
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "_collect_calib"):
        raise RuntimeError("Policy model does not support calibration capture.")

    model._collect_calib = True
    model._calib_buffer.clear()

    try:
        while len(model._calib_buffer) < target_samples:
            worker.sample()
    finally:
        model._collect_calib = False

    return model._calib_buffer[:target_samples]


def maybe_prepare_trt_calibration(compressors, trainer, hparams):
    """If TensorRT INT8 is requested and no calibration is set, collect once before training."""
    try:
        trt_target = hparams.get("quant_trt_calib_batches", 4) * hparams.get("quant_trt_calib_batch_size", 64)
        for comp in compressors:
            if isinstance(comp, QuantCompressor) and comp.mode == "tensorrt_int8" and comp.calibration_data is None:
                print(f"Collecting {trt_target} calibration samples for TensorRT INT8...")
                calib = _collect_trt_calibration(trainer.algo, trt_target, hparams.get("quant_trt_calib_batch_size", 64))
                if calib:
                    comp.set_calibration_data(calib)
                    print(f"Calibration data prepared for TensorRT INT8.")
                else:
                    print(f"Failed to collect calibration samples; TensorRT INT8 may fail.")
                break
    except Exception as exc:
        print(f"Calibration data collection failed: {exc}")


if __name__ == "__main__":

    ray.init(include_dashboard=False)

    hparams = DEFAULT_HPARAMS
    apply_global_seed(hparams.get("seed"))
    if hparams.get("seed") is not None:
        print(f"[main] Using seed: {hparams['seed']}")
    hidden_layers = [hparams["hidden_dim"]] * hparams["hidden_depth"]

    for exp in EXPERIMENTS:
        exp_device = resolve_device(exp.get("device", "cpu"))
        print(f"\n========== Running {exp['name']} ({exp['mode'].value}) ==========")
        print(f"[main] Using device: {exp_device}")
        config = build_config(hidden_layers, exp_device, hparams)
        compressors = build_compressors(exp, exp_device, hparams)
        infer_index = exp.get("infer_output_index")
        if compressors:
            if infer_index is None or infer_index < 0:
                infer_index = len(compressors) - 1
        else:
            infer_index = -1

        trainer = Trainer(
            config=config,
            compressors=compressors,
            compile_mode=exp["mode"],
            trigger_every=exp.get("trigger_every", 0),
            enable_diff_check=exp.get("enable_diff_check", True),
            compile_training_backbone=exp["compile_training_backbone"],
            log_dir=os.path.join("logs", exp["name"]),
            device=exp_device,
            infer_output_index=infer_index,
            wandb_enabled=hparams.get("use_wandb", False),
            wandb_project=hparams.get("wandb_project"),
            wandb_run_name=f"{exp['name']}_{hparams['hidden_dim']}_{exp_device}_{exp['trigger_every']}",
            wandb_config={
                "experiment": exp["name"],
                "env_id": hparams["env_id"],
                "device": exp_device,
                "group": hparams.get("wandb_group"),
            },
            async_warmup=exp.get("async_warmup", False),
            min_epoch_before_compress=exp.get(
                "min_epoch_before_compress",
                hparams.get("min_epoch_before_compress", 0),
            ),
        )
        # If using TensorRT int8 quant, collect calibration data before training starts
        maybe_prepare_trt_calibration(compressors, trainer, hparams)

        trainer.run(num_epochs=hparams["num_epochs"])
        trainer.summary()

    ray.shutdown()
