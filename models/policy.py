# path: models/policy.py

import time

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
        device_str = custom_conf.get("device", "cpu")
        try:
            resolved_device = torch.device(device_str)
            if resolved_device.type.startswith("cuda") and not torch.cuda.is_available():
                print(f"[CustomPolicyNet] ⚠️ Device {device_str} unavailable, fallback to CPU.")
                resolved_device = torch.device("cpu")
        except (RuntimeError, TypeError):
            print(f"[CustomPolicyNet] ⚠️ Invalid device {device_str}, fallback to CPU.")
            resolved_device = torch.device("cpu")
        self.device = resolved_device

        # === 未压缩的训练用 backbone ===
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        self.backbone = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual).to(self.device)

        # === 可选：压缩后的推理 backbone（由 PolicyManager 注入）===
        self.__dict__["_compiled_backbone"] = None
        self.use_compiled = False

        # value_function 输出缓存
        self._value_out = None
        self._inference_time_accum = 0.0

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
        compiled_bb = getattr(self, "_compiled_backbone", None)
        bb = compiled_bb if (self.use_compiled and compiled_bb is not None) else self.backbone
        t0 = time.perf_counter()
        if bb is not None:
            # 将观测移动到 backbone 所在设备，避免 CPU/GPU 混用
            try:
                device = next(bb.parameters()).device
            except StopIteration:
                device = obs.device
            obs = obs.to(device)

        logits, value = bb(obs)
        self._value_out = value.view(-1)     # RLlib 需要 [B] 向量
        self._inference_time_accum += (time.perf_counter() - t0)

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
        if "compiled_backbone" in self._modules:
            self._modules.pop("compiled_backbone")
        self.__dict__["_compiled_backbone"] = compiled_bb
        self.use_compiled = (compiled_bb is not None)

    def warmup_compiled_backbone(self, batch_size: int = 32):
        """通过一次 dummy 前向触发 torch.compile 的图捕获，避免首轮延迟。"""
        compiled_bb = getattr(self, "_compiled_backbone", None)
        if not self.use_compiled or compiled_bb is None:
            return
        in_dim = getattr(self, "in_dim", None)
        if in_dim is None:
            return
        try:
            device = next(compiled_bb.parameters()).device
        except StopIteration:
            device = self.device
        dummy_obs = torch.randn(batch_size, in_dim, device=device)
        with torch.no_grad():
            compiled_bb(dummy_obs)

    def update_compiled_backbone_weights(self, state_dict):
        compiled_bb = getattr(self, "_compiled_backbone", None)
        if compiled_bb is None or state_dict is None:
            return
        target = getattr(compiled_bb, "_orig_mod", compiled_bb)
        params = list(target.parameters())
        if params:
            device = params[0].device
        else:
            device = self.device
        converted = {}
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                converted[k] = v.to(device)
            else:
                converted[k] = v
        target.load_state_dict(converted, strict=False)

    def consume_inference_time(self) -> float:
        total = self._inference_time_accum
        self._inference_time_accum = 0.0
        return total

    # ------------------------------------------------------------
    # state_dict/load_state_dict：处理 torch.compile 引入的 _orig_mod 前缀
    # ------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state = self._strip_compiled_prefix(state)
        state = self._remove_compiled_backbone(state)
        return state

    def load_state_dict(self, state_dict, strict=True):
        filtered = self._remove_compiled_backbone(state_dict)
        adjusted = self._apply_compiled_prefix(filtered)
        return super().load_state_dict(adjusted, strict=strict)

    @staticmethod
    def _strip_compiled_prefix(state):
        if state is None:
            return state
        compiled_prefix = "backbone._orig_mod."
        if not any(k.startswith(compiled_prefix) for k in state.keys()):
            return state
        cleaned = state.__class__()
        for k, v in state.items():
            if k.startswith(compiled_prefix):
                new_key = "backbone." + k[len(compiled_prefix):]
                cleaned[new_key] = v
            else:
                cleaned[k] = v
        return cleaned

    @staticmethod
    def _remove_compiled_backbone(state):
        if state is None:
            return state
        compiled_prefix = "compiled_backbone."
        # always filter out compiled_backbone keys so policy state_dict stays clean
        cleaned = state.__class__()
        for k, v in state.items():
            if k.startswith(compiled_prefix):
                continue
            cleaned[k] = v
        return cleaned

    def _apply_compiled_prefix(self, state):
        if state is None:
            return state
        needs_prefix = hasattr(self.backbone, "_orig_mod")
        compiled_prefix = "backbone._orig_mod."
        plain_prefix = "backbone."
        adjusted = state.__class__()
        for k, v in state.items():
            if k.startswith(plain_prefix) and needs_prefix:
                new_key = compiled_prefix + k[len(plain_prefix):]
                adjusted[new_key] = v
            else:
                adjusted[k] = v
        return adjusted


# 注册 RLlib model
ModelCatalog.register_custom_model("custom_policy", CustomPolicyNet)
