"""ACT's action chunk in numpy, from the raw LeRobot checkpoint, against test-act.

Follows LeRobot's ACTPolicy.select_action at inference: the inputs normalized
with the checkpoint's statistics, ResNet-18 to layer4, the latent token at zero,
a 4-layer post-norm encoder, the 1-layer decoder from zeros, and the action
un-normalized. Nothing here reuses the vAi loader's folding, so the decoder's
precomputed queries are checked as well.

    test-act > out.txt
    python tests/act/reference.py out.txt
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "data" / "ckpts" / "act" / "lerobot" / "transfer-cube" / "model.safetensors"
H, W, D, HEADS, CHUNK = 480, 640, 512, 8, 100


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        body = f.read()
    header.pop("__metadata__", None)
    out = {}
    for key, info in header.items():
        assert info["dtype"] == "F32", key
        a, b = info["data_offsets"]
        out[key] = np.frombuffer(body[a:b], dtype=np.float32).reshape(info["shape"]).astype(np.float64)
    return out


def conv2d(x: np.ndarray, w: np.ndarray, stride: int) -> np.ndarray:
    """x [C, H, W], w [O, C, K, K], padding K // 2, as torchvision's ResNet pads."""
    k = w.shape[2]
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))[:, ::stride, ::stride]
    c, ho, wo = win.shape[:3]
    cols = win.transpose(1, 2, 0, 3, 4).reshape(ho * wo, c * k * k)
    return (cols @ w.reshape(w.shape[0], -1).T).T.reshape(w.shape[0], ho, wo)


def batch_norm(x: np.ndarray, sd: dict, key: str) -> np.ndarray:
    scale = sd[key + ".weight"] / np.sqrt(sd[key + ".running_var"] + 1e-5)
    shift = sd[key + ".bias"] - sd[key + ".running_mean"] * scale
    return x * scale[:, None, None] + shift[:, None, None]


def max_pool(x: np.ndarray) -> np.ndarray:
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))[:, ::2, ::2]
    return win.max(axis=(3, 4))


def resnet18(x: np.ndarray, sd: dict) -> np.ndarray:
    r = "model.backbone."
    x = np.maximum(batch_norm(conv2d(x, sd[r + "conv1.weight"], 2), sd, r + "bn1"), 0)
    x = max_pool(x)
    for layer, stride in zip(range(1, 5), (1, 2, 2, 2)):
        for block in range(2):
            p = f"{r}layer{layer}.{block}."
            s = stride if block == 0 else 1
            y = np.maximum(batch_norm(conv2d(x, sd[p + "conv1.weight"], s), sd, p + "bn1"), 0)
            y = batch_norm(conv2d(y, sd[p + "conv2.weight"], 1), sd, p + "bn2")
            if p + "downsample.0.weight" in sd:
                x = batch_norm(conv2d(x, sd[p + "downsample.0.weight"], s), sd, p + "downsample.1")
            x = np.maximum(y + x, 0)
    return x


def sine_2d(h: int, w: int, dim: int) -> np.ndarray:
    """ACTSinusoidalPositionEmbedding2d for a [., ., h, w] map, as [h*w, dim]."""
    half = dim // 2
    y = np.arange(1, h + 1, dtype=np.float64)[:, None].repeat(w, 1) / (h + 1e-6) * 2 * math.pi
    x = np.arange(1, w + 1, dtype=np.float64)[None, :].repeat(h, 0) / (w + 1e-6) * 2 * math.pi
    freq = 10000.0 ** (2 * (np.arange(half) // 2) / half)
    def embed(v):
        r = v[..., None] / freq
        out = np.empty_like(r)
        out[..., 0::2] = np.sin(r[..., 0::2])
        out[..., 1::2] = np.cos(r[..., 1::2])
        return out
    return np.concatenate([embed(y), embed(x)], axis=-1).reshape(h * w, dim)


def layer_norm(x: np.ndarray, sd: dict, key: str) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * sd[key + ".weight"] + sd[key + ".bias"]


def mha(q: np.ndarray, k: np.ndarray, v: np.ndarray, sd: dict, key: str) -> np.ndarray:
    w, b = sd[key + ".in_proj_weight"], sd[key + ".in_proj_bias"]
    Q = q @ w[:D].T + b[:D]
    K = k @ w[D:2 * D].T + b[D:2 * D]
    V = v @ w[2 * D:].T + b[2 * D:]
    dh = D // HEADS
    split = lambda m: m.reshape(m.shape[0], HEADS, dh).transpose(1, 0, 2)
    s = split(Q) @ split(K).transpose(0, 2, 1) / math.sqrt(dh)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = (s @ split(V)).transpose(1, 0, 2).reshape(Q.shape[0], D)
    return o @ sd[key + ".out_proj.weight"].T + sd[key + ".out_proj.bias"]


def ffn(x: np.ndarray, sd: dict, key: str) -> np.ndarray:
    h = np.maximum(x @ sd[key + "linear1.weight"].T + sd[key + "linear1.bias"], 0)
    return h @ sd[key + "linear2.weight"].T + sd[key + "linear2.bias"]


def forward(sd: dict, image: np.ndarray, state: np.ndarray) -> np.ndarray:
    eps = 1e-8
    image = (image - sd["normalize_inputs.buffer_observation_images_top.mean"]) \
        / (sd["normalize_inputs.buffer_observation_images_top.std"] + eps)
    state = (state - sd["normalize_inputs.buffer_observation_state.mean"]) \
        / (sd["normalize_inputs.buffer_observation_state.std"] + eps)

    feat = resnet18(image, sd)
    feat = conv2d(feat, sd["model.encoder_img_feat_input_proj.weight"], 1) \
        + sd["model.encoder_img_feat_input_proj.bias"][:, None, None]
    img_tokens = feat.reshape(D, -1).T

    latent = np.zeros(32) @ sd["model.encoder_latent_input_proj.weight"].T + sd["model.encoder_latent_input_proj.bias"]
    state_tok = state @ sd["model.encoder_robot_state_input_proj.weight"].T + sd["model.encoder_robot_state_input_proj.bias"]
    x = np.concatenate([latent[None], state_tok[None], img_tokens])
    pos = np.concatenate([sd["model.encoder_1d_feature_pos_embed.weight"], sine_2d(feat.shape[1], feat.shape[2], D)])

    for i in range(4):
        p = f"model.encoder.layers.{i}."
        x = layer_norm(x + mha(x + pos, x + pos, x, sd, p + "self_attn"), sd, p + "norm1")
        x = layer_norm(x + ffn(x, sd, p), sd, p + "norm2")
    memory = x

    qpos = sd["model.decoder_pos_embed.weight"]
    p = "model.decoder.layers.0."
    y = np.zeros((CHUNK, D))
    y = layer_norm(y + mha(y + qpos, y + qpos, y, sd, p + "self_attn"), sd, p + "norm1")
    y = layer_norm(y + mha(y + qpos, memory + pos, memory, sd, p + "multihead_attn"), sd, p + "norm2")
    y = layer_norm(y + ffn(y, sd, p), sd, p + "norm3")
    y = layer_norm(y, sd, "model.decoder.norm")

    a = y @ sd["model.action_head.weight"].T + sd["model.action_head.bias"]
    return a * sd["unnormalize_outputs.buffer_action.std"] + sd["unnormalize_outputs.buffer_action.mean"]


def main() -> None:
    sd = load(CKPT)
    i = np.arange(H, dtype=np.float32)[None, :, None]
    j = np.arange(W, dtype=np.float32)[None, None, :]
    c = np.arange(3, dtype=np.float32)[:, None, None]
    image = (0.5 + 0.5 * np.sin(np.float32(0.05) * i + np.float32(0.07) * j + c)).astype(np.float64)
    state = (0.1 * np.sin(np.arange(14, dtype=np.float32))).astype(np.float64)

    ref = forward(sd, image, state)
    got = np.loadtxt(sys.argv[1])
    assert got.shape == ref.shape, (got.shape, ref.shape)

    err = np.abs(got - ref)
    scale = np.abs(ref).max()
    print(f"reference |a| max {scale:.4f}")
    print(f"abs error  max {err.max():.3e}  median {np.median(err):.3e}")
    print(f"rel error  max {err.max() / scale:.3e}  (to the largest |a|)")
    print("first row  ref", np.array2string(ref[0, :6], precision=4))
    print("first row  vAi", np.array2string(got[0, :6], precision=4))


if __name__ == "__main__":
    main()
