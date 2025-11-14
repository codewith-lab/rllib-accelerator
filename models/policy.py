# path: models/policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


# ============================================================
# 1. PolicyBackbone（纯 PyTorch MLP）——可被 torch.compile/quant/prune
# ============================================================
class PolicyBackbone(nn.Module):
    """
    纯 PyTorch 前向骨干，用于被压缩（compile/quant/prune/distill）。
    返回 logits 和 value。
    """

    def __init__(self, in_dim: int, num_outputs: int, hidden_dims=None, use_residual: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if len(hidden_dims) == 0:
            hidden_dims = [64]

        self.hidden_layers = nn.ModuleList()
        prev = in_dim
        for dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev, dim))
            prev = dim

        self.policy_head = nn.Linear(prev, num_outputs)
        self.value_head = nn.Linear(prev, 1)
        self.use_residual = use_residual

    def forward(self, obs: torch.Tensor):
        x = obs
        for layer in self.hidden_layers:
            use_skip = self.use_residual and getattr(layer, "in_features", None) == getattr(layer, "out_features", None)
            residual = x if use_skip else None
            x = F.relu(layer(x))
            if residual is not None:
                x = x + residual

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value    # value: [B, 1]


# ============================================================
# 2. RLlib 的 CustomPolicyNet
#    - 训练时使用 self.backbone（未压缩）
#    - 推理时可切换到 self.compiled_backbone
# ============================================================
class CustomPolicyNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space,
                 num_outputs, model_config, name):

        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_dim = obs_space.shape[0]
        self.in_dim = in_dim
        self.num_outputs = num_outputs

        hidden_dims = model_config.get("fcnet_hiddens", [64, 64])
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if len(hidden_dims) == 0:
            hidden_dims = [64, 64]
        custom_conf = model_config.get("custom_model_config", {})
        use_residual = bool(custom_conf.get("use_residual", False))

        # === 未压缩的训练用 backbone ===
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        self.backbone = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual)

        # === 可选：压缩后的推理 backbone（由 PolicyManager 注入）===
        self.compiled_backbone = None
        self.use_compiled = False

        # value_function 输出缓存
        self._value_out = None

    # ------------------------------------------------------------
    # RLlib forward
    # ------------------------------------------------------------
    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()

        # 选择训练 or 推理 backbone
        bb = (
            self.compiled_backbone
            if (self.use_compiled and self.compiled_backbone is not None)
            else self.backbone
        )
        if bb is not None:
            # 将观测移动到 backbone 所在设备，避免 CPU/GPU 混用
            try:
                device = next(bb.parameters()).device
            except StopIteration:
                device = obs.device
            obs = obs.to(device)

        logits, value = bb(obs)
        self._value_out = value.view(-1)     # RLlib 需要 [B] 向量

        return logits, state

    # ------------------------------------------------------------
    # RLlib 需要 value_function()
    # ------------------------------------------------------------
    def value_function(self):
        return self._value_out

    # ------------------------------------------------------------
    # PolicyManager 用于给 sampler 注入新的推理模型
    # ------------------------------------------------------------
    def set_compiled_backbone(self, compiled_bb: nn.Module):
        """在 sampler worker 上切换推理 backbone。"""
        self.compiled_backbone = compiled_bb
        self.use_compiled = (compiled_bb is not None)


# 注册 RLlib model
ModelCatalog.register_custom_model("custom_policy", CustomPolicyNet)
