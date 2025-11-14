# path: main.py

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from framework.trainer import Trainer
from framework.policy_manager import CompileMode
from compression.compile_compressor import CompileCompressor

# 注册全局模型
from models.policy import CustomPolicyNet    # noqa


if __name__ == "__main__":

    # ------------------------------------------------------------
    # 初始化 Ray
    # ------------------------------------------------------------
    ray.init(include_dashboard=False)

    hidden_dim = 1024
    hidden_depth = 8
    hidden_layers = [hidden_dim] * hidden_depth
    compile_training_backbone = False  # 设置为 True 可同步编译训练 backbone

    # ------------------------------------------------------------
    # RLlib 配置
    # ------------------------------------------------------------
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .training(
            model={
                "custom_model": "custom_policy",
                "fcnet_hiddens": hidden_layers,
                "custom_model_config": {"use_residual": True},
            },
            train_batch_size=80000,
            lr=1e-4,
        )
        .rollouts(num_rollout_workers=4, rollout_fragment_length=20000)
    )

    # ------------------------------------------------------------
    # 设置压缩器（你也可以加入 QuantCompressor, PruneCompressor）
    # ------------------------------------------------------------
    compressors = [
        CompileCompressor(backend="inductor", diff_threshold=1e-4),
    ]

    # ------------------------------------------------------------
    # 创建训练器（同步 OR 异步）
    # ------------------------------------------------------------
    trainer = Trainer(
        config=config,
        compressors=compressors,
        compile_mode=CompileMode.NONE,   # 可改 SYNC / NONE / ASYNC
        trigger_every=1,                 # 固定周期触发
        enable_diff_check=False,         # 启用差分检查
        compile_training_backbone=compile_training_backbone,
        log_dir="logs"
    )

    # ------------------------------------------------------------
    # 运行训练
    # ------------------------------------------------------------
    trainer.run(num_epochs=10)
    trainer.summary()

    # ------------------------------------------------------------
    # 关闭 Ray
    # ------------------------------------------------------------
    ray.shutdown()
