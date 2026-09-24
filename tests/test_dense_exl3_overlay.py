#!/usr/bin/env python3
"""CPU checks for overlay/exl3.py [dense-exl3]: config parsing, module
matching, per-shard TP geometry for TP2/TP4 (incl. replicated and bf16
shards), conflict refusals, codebook-marker validation, and the ABLIT
EXL3-o_proj refusal. Runs with real torch and stubbed vllm modules — it never
imports exllamav3 (marker checks raise before LinearEXL3 construction).

."""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXL3_SRC = ROOT / "overlay" / "exl3.py"
ABLIT_SRC = ROOT / "overlay" / "ablit_runtime.py"

MLA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
LAYER_TYPES = [
    "linear_attention" if i not in MLA_LAYERS else "deepseek_sparse_attention"
    for i in range(45)
]

# GLM-5.3-Flash dims (config.json + image model code): hidden 4096, dense MLP
# intermediate 12288 (layers 0-2), shared experts 2048 (= moe_intermediate x 1),
# 64 attention heads x 256 head dims (MLA), KDA 64 heads x 128.
H = 4096
DENSE_I = 12288
SHARED_I = 2048
KDA_HEADS, KDA_HD = 64, 128
MLA_HEAD_DIMS = 64 * 256


class _Cfg:
    def __init__(self, layer_types):
        class _HF:
            pass

        class _MC:
            hf_text_config = _HF()

        self.model_config = _MC()
        self.model_config.hf_text_config.layer_types = layer_types


def _install_vllm_stubs(layer_types=None):
    """Real classes for isinstance dispatch; everything else is inert."""
    linear = types.ModuleType("vllm.model_executor.layers.linear")

    class LinearBase(torch.nn.Module):
        pass

    class RowParallelLinear(LinearBase):
        pass

    class MergedColumnParallelLinear(LinearBase):
        pass

    class QKVParallelLinear(LinearBase):
        pass

    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            raise AssertionError("stock apply must not run in these tests")

    class LinearMethodBase:
        pass

    for name, obj in vars().items():
        setattr(linear, name, obj)

    fm_base = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe_method_base")

    class FusedMoEMethodBase:
        pass

    fm_base.FusedMoEMethodBase = FusedMoEMethodBase

    class RoutedExperts(torch.nn.Module):
        pass

    routed = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")
    routed.RoutedExperts = RoutedExperts

    qc = types.ModuleType("vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        def is_layer_free(self, layer):  # pragma: no cover - unused here
            return True

    qc.QuantizationConfig = QuantizationConfig

    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: logging.getLogger(name)
    vllm_config_mod = types.ModuleType("vllm.config")
    vllm_config_mod.get_current_vllm_config = lambda: _Cfg(layer_types or LAYER_TYPES)
    registered = {}

    def register(name):
        def deco(cls):
            registered[name] = cls
            return cls

        return deco

    vllm_quant = types.ModuleType("vllm.model_executor.layers.quantization")
    vllm_quant.register_quantization_config = register
    vllm_utils = types.ModuleType("vllm.model_executor.utils")
    vllm_utils.set_weight_attrs = lambda weight, attrs: [
        setattr(weight, k, v) for k, v in (attrs or {}).items()
    ]
    fm_cfg = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    fm_cfg.FusedMoEQuantConfig = object

    for mod in (vllm, vllm_logger, vllm_config_mod, vllm_quant, vllm_utils, fm_cfg):
        sys.modules.setdefault(mod.__name__, mod)
    for name, mod in (
        ("vllm.logger", vllm_logger),
        ("vllm.config", vllm_config_mod),
        ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
        ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
        ("vllm.model_executor.layers.linear", linear),
        ("vllm.model_executor.layers.fused_moe", types.ModuleType("vllm.model_executor.layers.fused_moe")),
        ("vllm.model_executor.layers.fused_moe.fused_moe_method_base", fm_base),
        ("vllm.model_executor.layers.fused_moe.routed_experts", routed),
        ("vllm.model_executor.layers.quantization", vllm_quant),
        ("vllm.model_executor.layers.quantization.base_config", qc),
        ("vllm.model_executor.utils", vllm_utils),
        ("vllm.model_executor.layers.fused_moe.config", fm_cfg),
    ):
        sys.modules[name] = mod
    return dict(
        LinearBase=LinearBase,
        RowParallelLinear=RowParallelLinear,
        MergedColumnParallelLinear=MergedColumnParallelLinear,
        QKVParallelLinear=QKVParallelLinear,
        UnquantizedLinearMethod=UnquantizedLinearMethod,
        registered=registered,
    )


def _load_exl3():
    spec = importlib.util.spec_from_file_location("exl3_dense_test", EXL3_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nr_config(mod, layers=None, base_bits=4):
    return mod.Exl3Config(bits=base_bits, non_routed_exl3={"layers": layers or {}})


def config_tests(mod):
    C = mod.Exl3Config
    # from_config round-trip: the pack's non_routed_exl3 block flows through
    # the kwargs passthrough exactly as dense_overlay.py writes it.
    cfg = C.from_config({
        "quant_method": "exl3", "bits": 4, "codebook": "mcg",
        "scope": "glm53_routed_experts_only",
        "tensor_storage": {"big": "ledger"},
        "non_routed_exl3": {
            "codebook": "mul1",
            "layers": {
                "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a": {
                    "bits": 6, "bf16_shards": [3, 4, 5]},
                "language_model.model.layers.0.mlp.gate_up_proj": {"bits": 5},
            },
        },
    })
    assert "tensor_storage" not in cfg.raw_config
    m = cfg._matches_non_routed_exl3
    assert m("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a")
    assert m("language_model.model.layers.0.mlp.gate_up_proj")
    assert not m("language_model.model.layers.0.self_attn.o_proj")
    assert not m("language_model.model.layers.0.self_attn.f_b_proj")
    assert not m("language_model.model.layers.3.self_attn.kv_b_proj")
    assert cfg._bits_for_non_routed("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a") == 6
    assert cfg._bits_for_non_routed("language_model.model.layers.0.mlp.gate_up_proj") == 5
    assert cfg._bf16_shards_for("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a") == [3, 4, 5]
    assert cfg._bf16_shards_for("language_model.model.layers.0.mlp.gate_up_proj") == []

    # A config without the block matches nothing and stays stock.
    stock = C.from_config({"quant_method": "exl3", "bits": 4})
    assert not stock.non_routed_exl3
    assert not stock._matches_non_routed_exl3("language_model.model.layers.0.mlp.down_proj")

    # bf16_shards must arrive sorted and contiguous (tail checked at create).
    for bad in ({"layers": {"a.b": {"bits": 6, "bf16_shards": [4, 3, 5]}}},
                {"layers": {"a.b": {"bits": 6, "bf16_shards": [3, 5]}}}):
        try:
            _nr_config(mod, **bad)
        except ValueError as exc:
            assert "bf16_shards" in str(exc), exc
        else:
            raise AssertionError(f"bad bf16_shards accepted: {bad}")

    # B2: declared-but-never-constructed modules refuse loudly
    declared = {"x.self_attn.o_proj": {"bits": 6}, "y.mlp.down_proj": {"bits": 5}}
    cfg_b2 = _nr_config(mod, layers=declared)
    mod.Exl3LinearMethod(cfg_b2, "x.self_attn.o_proj", bits=6)
    try:
        cfg_b2._assert_non_routed_built()
    except RuntimeError as exc:
        assert "y.mlp.down_proj" in str(exc) and "1/2" in str(exc), exc
    else:
        raise AssertionError("unbuilt declared module must refuse")
    empty = _nr_config(mod, layers={})
    empty._assert_non_routed_built()  # no declarations -> no-op

    for bad in ({"layers": {"a.b": {"bits": 7}}}, {"layers": {"a.b": {"bits": 1}}},
                {"layers": {"a.b": {}}}):
        try:
            _nr_config(mod, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad non_routed bits accepted: {bad}")
    # kv_b_proj is refused wherever it is declared (MLA reads the weight).
    try:
        _nr_config(mod, layers={"x.self_attn.kv_b_proj": {"bits": 6}})
    except ValueError as exc:
        assert "kv_b_proj" in str(exc)
    else:
        raise AssertionError("kv_b_proj accepted")
    print("config parsing OK")


def _fake_layer(cls, prefix, tp_rank, tp_size, out_sizes, in_size, replicated=()):
    layer = cls()
    layer.prefix = prefix
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.replicated_shard_ids = tuple(replicated)
    layer.output_partition_sizes = list(out_sizes)
    layer.input_size_per_partition = in_size
    return layer


def _pack(mod, k, out_rows, in_cols, marker="mul1", seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(
        -30000, 30000, (in_cols // 16, out_rows // 16, 16 * k),
        dtype=torch.int16, generator=g)
    suh = torch.randn(in_cols, dtype=torch.float16, generator=g)
    svh = torch.randn(out_rows, dtype=torch.float16, generator=g)
    mk = torch.tensor([mod.MUL1_MARKER_SIGNED_INT32 if marker == "mul1"
                       else mod.MCG_MARKER_SIGNED_INT32], dtype=torch.int32)
    return t, suh, svh, mk, ("mcg" if marker == "mcg" else "mul1")


def _load_all(mod, method, layer, shard_tensor_map, shard_ids):
    """Feed the loaders exactly what the pack ships: packed parts only for
    EXL3 shards, `weight` staging for every shard (EXL3 ones get discarded)."""
    for suffix in ("trellis", "suh", "svh"):
        p = getattr(layer, suffix)
        for idx, sid in enumerate(shard_ids):
            if (idx, suffix) not in shard_tensor_map:
                continue
            p.weight_loader(p, shard_tensor_map[(idx, suffix)], sid)
    for idx, sid in enumerate(shard_ids):
        if (idx, "marker") not in shard_tensor_map:
            continue
        # the pack names the marker tensor *.mul1 or *.mcg; only that param
        # receives it (the other stays zero)
        p = getattr(layer, shard_tensor_map[(idx, "marker_suffix")])
        p.weight_loader(p, shard_tensor_map[(idx, "marker")], sid)
    if method.quant_config._bf16_shards_for(layer.prefix):
        w = layer.weight
        for idx, sid in enumerate(shard_ids):
            if (idx, "weight") not in shard_tensor_map:
                continue
            w.weight_loader(w, shard_tensor_map[(idx, "weight")], sid)


def geometry_tests(mod, stubs):
    Exl3LinearMethod = mod.Exl3LinearMethod
    
    def method_for(prefix, bits=6, bf16=None):
        c = _nr_config(mod, layers={prefix: {"bits": bits, **({"bf16_shards": bf16} if bf16 else {})}})
        return Exl3LinearMethod(c, prefix, bits=bits)

    # ---- MLA o_proj (RowParallel, input sharded) at TP2 and TP4 -----------
    for tp in (2, 4):
        methods = []
        layers = []
        for rank in range(tp):
            m = method_for("language_model.model.layers.3.self_attn.o_proj")
            layer = _fake_layer(
                stubs["RowParallelLinear"],
                "language_model.model.layers.3.self_attn.o_proj",
                rank, tp, [H], MLA_HEAD_DIMS // tp)
            m.create_weights(layer, MLA_HEAD_DIMS // tp, [H], MLA_HEAD_DIMS, H, torch.bfloat16)
            methods.append(m)
            layers.append(layer)
            assert layer.trellis.shape == (MLA_HEAD_DIMS // tp // 16, H // 16, 96)
            assert layer.suh.shape == (1, MLA_HEAD_DIMS // tp)
            assert layer.svh.shape == (H,)
        full_t, full_suh, full_svh, mk, mkind = _pack(mod, 6, H, MLA_HEAD_DIMS, seed=1)
        for rank in range(tp):
            _load_all(mod, methods[rank], layers[rank],
                      {(0, "trellis"): full_t, (0, "suh"): full_suh,
                       (0, "svh"): full_svh, (0, "marker"): mk,
                       (0, "marker_suffix"): mkind},
                      [None])
            assert layers[rank].svh.tolist() == full_svh.tolist()
        # Row-parallel trellis/suh narrow on dim 0: ranks tile the rows.
        joined_t = torch.cat([l.trellis for l in layers], dim=0)
        assert torch.equal(joined_t, full_t)
        joined_suh = torch.cat([l.suh[:, :].squeeze(0) for l in layers], dim=0)
        assert torch.equal(joined_suh, full_suh)
    print("MLA o_proj row-parallel TP2/TP4 OK")

    # ---- MLA q_b_proj (ColumnParallel, output sharded) at TP2/TP4 ---------
    for tp in (2, 4):
        out_local = MLA_HEAD_DIMS // tp
        layers = []
        methods = []
        for rank in range(tp):
            m = method_for("language_model.model.layers.3.self_attn.q_b_proj")
            layer = _fake_layer(
                stubs["LinearBase"],  # plain ColumnParallel path
                "language_model.model.layers.3.self_attn.q_b_proj",
                rank, tp, [out_local], 1536)
            m.create_weights(layer, 1536, [out_local], MLA_HEAD_DIMS, MLA_HEAD_DIMS, torch.bfloat16)
            layers.append(layer)
            methods.append(m)
        full_t, full_suh, full_svh, mk, mkind = _pack(mod, 6, MLA_HEAD_DIMS, 1536, seed=2)
        for rank in range(tp):
            _load_all(mod, methods[rank], layers[rank],
                      {(0, "trellis"): full_t, (0, "suh"): full_suh,
                       (0, "svh"): full_svh, (0, "marker"): mk,
                       (0, "marker_suffix"): mkind}, [None])
            assert torch.equal(layers[rank].suh[0], full_suh)  # suh not sharded
        joined_t = torch.cat([l.trellis for l in layers], dim=1)
        assert torch.equal(joined_t, full_t)
        joined_svh = torch.cat([l.svh for l in layers], dim=0)
        assert torch.equal(joined_svh, full_svh)
    print("MLA q_b_proj column-parallel TP2/TP4 OK")

    # ---- KDA in_proj_qkvbfg_a: merged shards, q/k/v EXL3, b/f_a/g_a bf16,
    # f_a/g_a replicated. Shard ids are ints 0..5 from stacked_params_mapping.
    for tp in (2, 4):
        P = KDA_HEADS * KDA_HD
        sizes = [P // tp, P // tp, P // tp, KDA_HEADS // tp, KDA_HD, KDA_HD]
        full_sizes = [P, P, P, KDA_HEADS // 1, KDA_HD, KDA_HD]
        # full b shard rows: KDA_HEADS (not divided by tp in the checkpoint
        # module contract: b is divided; the loader slices per-rank rows).
        methods, layer_sets = [], []
        for rank in range(tp):
            m = method_for(
                "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
                bf16=[3, 4, 5])
            layer = _fake_layer(
                stubs["MergedColumnParallelLinear"],
                "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
                rank, tp, sizes, H, replicated=(4, 5))
            m.create_weights(layer, H, sizes, H, sum(full_sizes), torch.bfloat16)
            methods.append(m)
            layer_sets.append(layer)
            exl_tiles = sum(s // 16 for i, s in enumerate(sizes) if i not in (3, 4, 5))
            assert layer.trellis.shape == (H // 16, exl_tiles, 96)
            assert layer.weight.shape == (
                KDA_HEADS // tp + 2 * KDA_HD, H)
        for rank in range(tp):
            layer = layer_sets[rank]
            tensors = {}
            for i in range(6):
                tensors[(i, "weight")] = torch.randn(full_sizes[i], H,
                                                     dtype=torch.bfloat16,
                                                     generator=torch.Generator().manual_seed(20 + i))
                if i in (3, 4, 5):
                    continue  # b/f_a/g_a are bf16 shards: no packed parts ship
                t, suh, svh, mk, mkind = _pack(mod, 6, full_sizes[i], H, seed=10 + i)
                tensors[(i, "trellis")] = t
                tensors[(i, "suh")] = suh
                tensors[(i, "svh")] = svh
                tensors[(i, "marker")] = mk
                tensors[(i, "marker_suffix")] = mkind
            _load_all(mod, methods[rank], layer, tensors, [0, 1, 2, 3, 4, 5])
            # markers: q/k/v carry mul1, bf16 shards carry nothing
            for i in range(3):
                assert layer.mul1[i, 0].item() == mod.MUL1_MARKER_SIGNED_INT32
                assert layer.mcg[i, 0].item() == 0
            for i in (3, 4, 5):
                assert layer.mul1[i, 0].item() == 0
                assert layer.mcg[i, 0].item() == 0
        # EXL3 shards tile the full q/k/v tensors across ranks.
        for i in range(3):
            joined_t = torch.cat([l.trellis[:, i * (P // tp // 16):(i + 1) * (P // tp // 16), :]
                                  for l in layer_sets], dim=1)
            full_t = tensors[(i, "trellis")]
            assert joined_t.shape == full_t.shape, (joined_t.shape, full_t.shape)
            # per-rank trellis columns == the rank's slice of the full tensor
            for rank in range(tp):
                cols = slice(rank * (P // tp // 16), (rank + 1) * (P // tp // 16))
                assert torch.equal(
                    layer_sets[rank].trellis[:, i * (P // tp // 16):(i + 1) * (P // tp // 16), :],
                    full_t[:, cols])
        # bf16 staging: q/k/v stale weights discarded; b/f_a/g_a rows staged.
        for rank in range(tp):
            w = layer_sets[rank].weight
            # b shard is column-parallel: rank rows are a contiguous slice
            # of the full KDA_HEADS rows.
            b_full = tensors[(3, "weight")]
            assert torch.equal(
                w[: KDA_HEADS // tp].to(torch.float32),
                b_full[rank * (KDA_HEADS // tp):(rank + 1) * (KDA_HEADS // tp)].to(torch.float32))
            # f_a/g_a replicated: full tensors on every rank.
            assert torch.equal(w[KDA_HEADS // tp: KDA_HEADS // tp + KDA_HD].to(torch.float32),
                               tensors[(4, "weight")].to(torch.float32))
            assert torch.equal(w[-KDA_HD:].to(torch.float32),
                               tensors[(5, "weight")].to(torch.float32))
    print("KDA in_proj merged shards + replicated + bf16 staging TP2/TP4 OK")

    # ---- dense MLP gate_up_proj at TP2/TP4 (merged, no bf16 shards) -------
    for tp in (2, 4):
        sizes = [DENSE_I // tp, DENSE_I // tp]
        layers = []
        methods = []
        for rank in range(tp):
            m = method_for("language_model.model.layers.1.mlp.gate_up_proj", bits=5)
            layer = _fake_layer(
                stubs["MergedColumnParallelLinear"],
                "language_model.model.layers.1.mlp.gate_up_proj",
                rank, tp, sizes, H)
            m.create_weights(layer, H, sizes, H, DENSE_I, torch.bfloat16)
            layers.append(layer)
            methods.append(m)
        gate_t, gate_suh, gate_svh, gate_mk, _gkind = _pack(mod, 5, DENSE_I, H, seed=30)
        up_t, up_suh, up_svh, up_mk, _ukind = _pack(mod, 5, DENSE_I, H, seed=31)
        for rank in range(tp):
            _load_all(mod, methods[rank], layers[rank],
                      {(0, "trellis"): gate_t, (0, "suh"): gate_suh,
                       (0, "svh"): gate_svh, (0, "marker"): gate_mk,
                       (0, "marker_suffix"): _gkind,
                       (1, "trellis"): up_t, (1, "suh"): up_suh,
                       (1, "svh"): up_svh, (1, "marker"): up_mk,
                       (1, "marker_suffix"): _ukind},
                      [0, 1])
            for i in range(2):
                assert layers[rank].mul1[i, 0].item() == mod.MUL1_MARKER_SIGNED_INT32
        for i, full_t in ((0, gate_t), (1, up_t)):
            cols = DENSE_I // tp // 16
            for rank in range(tp):
                got = layers[rank].trellis[:, i * cols:(i + 1) * cols, :]
                assert torch.equal(got, full_t[:, rank * cols:(rank + 1) * cols])
    print("dense MLP gate_up merged shards TP2/TP4 OK")

    # ---- shared experts: TP2 shards fine; TP3 divisibility refusal -------
    m = method_for("language_model.model.layers.5.mlp.shared_experts.gate_up_proj", bits=6)
    layer = _fake_layer(
        stubs["MergedColumnParallelLinear"],
        "language_model.model.layers.5.mlp.shared_experts.gate_up_proj",
        0, 2, [SHARED_I // 2, SHARED_I // 2], H)
    m.create_weights(layer, H, [SHARED_I // 2, SHARED_I // 2], H, SHARED_I, torch.bfloat16)
    full_t, _, _, mk, _ = _pack(mod, 6, SHARED_I, H, seed=40)
    p = layer.trellis
    p.weight_loader(p, full_t, 0)
    assert p.data.shape[1] == SHARED_I // 16
    # TP3: the loader's TP narrowing must refuse the indivisible width.
    m3 = method_for("language_model.model.layers.5.mlp.shared_experts.gate_up_proj", bits=6)
    layer3 = _fake_layer(
        stubs["MergedColumnParallelLinear"],
        "language_model.model.layers.5.mlp.shared_experts.gate_up_proj",
        0, 3, [SHARED_I // 2, SHARED_I // 2], H)  # sizes moot; refusal at load
    m3.create_weights(layer3, H, [SHARED_I // 2, SHARED_I // 2], H, SHARED_I, torch.bfloat16)
    try:
        layer3.trellis.weight_loader(layer3.trellis, full_t, 0)
    except ValueError as exc:
        assert "not divisible" in str(exc), exc
    else:
        raise AssertionError("TP3 shared-experts trellis must refuse (128 tiles % 3)")
    print("shared experts TP2 shard + TP3 divisibility refusal OK")

    # ---- trellis tile-width refusal (16-multiple) -------------------------
    m = method_for("x.self_attn.o_proj")
    layer = _fake_layer(
        stubs["RowParallelLinear"], "x.self_attn.o_proj", 0, 1, [4096], 4096 + 8)
    try:
        m.create_weights(layer, 4096 + 8, [4096], 4104, 4096, torch.bfloat16)
    except ValueError as exc:
        assert "16-wide" in str(exc), exc
    else:
        raise AssertionError("non-16-multiple input must refuse")
    # bf16 shards are exempt from the 16-multiple check (b=32/16 rows at TP2,
    # 16 at TP4; f_a/g_a 128) but a packed tensor for them still refuses.
    m = method_for("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
                   bf16=[3, 4, 5])
    layer = _fake_layer(
        stubs["MergedColumnParallelLinear"],
        "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
        0, 2, [4096, 4096, 4096, 32, 128, 128], H, replicated=(4, 5))
    m.create_weights(layer, H, [4096, 4096, 4096, 32, 128, 128], H, 12444,
                     torch.bfloat16)
    t3, s3, v3, mk3, _ = _pack(mod, 6, 32, H, seed=50)
    try:
        layer.trellis.weight_loader(layer.trellis, t3, 3)
    except RuntimeError as exc:
        assert "declared bf16" in str(exc), exc
    else:
        raise AssertionError("packed tensor for a bf16 shard must refuse")
    print("tile-width refusals OK")

    # ---- bf16 shards must be the shard tail (forward runs one tail GEMM) ---
    m = method_for("z.self_attn.in_proj_qkvbfg_a", bf16=[1, 2])
    layer = _fake_layer(
        stubs["MergedColumnParallelLinear"], "z.self_attn.in_proj_qkvbfg_a",
        0, 1, [1024, 1024, 1024, 64, 128, 128], H)
    try:
        m.create_weights(layer, H, [1024, 1024, 1024, 64, 128, 128], H, 3328,
                         torch.bfloat16)
    except ValueError as exc:
        assert "shard tail" in str(exc), exc
    else:
        raise AssertionError("non-tail bf16_shards must refuse")

    # ---- marker validation before any LinearEXL3 construction -------------
    m = method_for("language_model.model.layers.3.self_attn.o_proj")
    layer = _fake_layer(
        stubs["RowParallelLinear"], "language_model.model.layers.3.self_attn.o_proj",
        0, 1, [H], MLA_HEAD_DIMS)
    m.create_weights(layer, MLA_HEAD_DIMS, [H], MLA_HEAD_DIMS, H, torch.bfloat16)
    layer.mul1.data[0, 0] = mod.MUL1_MARKER_SIGNED_INT32
    layer.mcg.data[0, 0] = mod.MCG_MARKER_SIGNED_INT32  # both set -> refuse
    try:
        m.process_weights_after_loading(layer)
    except RuntimeError as exc:
        assert "exactly one codebook marker" in str(exc), exc
    else:
        raise AssertionError("both markers set must refuse")
    layer.mcg.data[0, 0] = 0
    layer.mul1.data[0, 0] = 12345  # set but wrong value -> refuse
    try:
        m.process_weights_after_loading(layer)
    except RuntimeError as exc:
        assert "bad mul1 marker" in str(exc), exc
    else:
        raise AssertionError("wrong marker value must refuse")
    print("codebook marker validation OK")


def dispatch_tests(mod, stubs):
    Exl3Config = mod.Exl3Config
    def set_env(**kw):
        for k in ("GLM53_DENSE_FP8", "GLM53_KDA_BF16_LARGE_M"):
            os.environ.pop(k, None)
        os.environ.update(kw)

    try:
        prefix = "language_model.model.layers.1.mlp.gate_up_proj"
        cfg = _nr_config(mod, layers={prefix: {"bits": 5}})
        set_env(GLM53_DENSE_FP8="off")
        method = cfg.get_quant_method(stubs["LinearBase"](), prefix)
        assert type(method).__name__ == "Exl3LinearMethod", method
        assert method.bits == 5

        set_env(GLM53_DENSE_FP8="dense")
        try:
            cfg.get_quant_method(stubs["LinearBase"](), prefix)
        except RuntimeError as exc:
            assert "GLM53_DENSE_FP8" in str(exc) and prefix in str(exc), exc
        else:
            raise AssertionError("DENSE_FP8 overlap must refuse")

        # LARGE_M is now supported on the EXL3 in_proj (load-time BF16 copy);
        # dispatch must return the EXL3 method, not refuse.
        kda_prefix = "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a"
        kda_cfg = _nr_config(mod, layers={kda_prefix: {"bits": 6, "bf16_shards": [3, 4, 5]}})
        set_env(GLM53_DENSE_FP8="off", GLM53_KDA_BF16_LARGE_M="1")
        method = kda_cfg.get_quant_method(stubs["LinearBase"](), kda_prefix)
        assert type(method).__name__ == "Exl3LinearMethod", method
        assert method.bits == 6

        # No pack block: DENSE_FP8 still dispatches its own groups, and a
        # non-matching prefix stays Unquantized.
        set_env(GLM53_DENSE_FP8="dense")
        stock = Exl3Config(bits=4)
        m = stock.get_quant_method(stubs["LinearBase"](),
                                   "language_model.model.layers.1.mlp.gate_up_proj")
        assert type(m).__name__ == "Glm53DenseFp8Method", m
        m = stock.get_quant_method(stubs["LinearBase"](),
                                   "language_model.model.layers.1.self_attn.q_conv1d")
        assert type(m).__name__ == "UnquantizedLinearMethod", m

        # Pack present, prefix not declared: stays stock even with groups off.
        set_env(GLM53_DENSE_FP8="off")
        m = cfg.get_quant_method(stubs["LinearBase"](),
                                 "language_model.model.layers.1.self_attn.o_proj")
        assert type(m).__name__ == "UnquantizedLinearMethod", m
    finally:
        for k in ("GLM53_DENSE_FP8", "GLM53_KDA_BF16_LARGE_M"):
            os.environ.pop(k, None)
    print("get_quant_method dispatch + conflict refusals OK")



def large_m_tests(mod):
    """GLM53_KDA_BF16_LARGE_M on the EXL3 KDA in_proj: retention keeps the
    dequantized q/k/v rows fp16 and the bf16 tail separate; the M>512 apply
    runs the custom op's exact arithmetic (fp16 GEMM -> x.dtype, then the
    bf16 tail GEMM), so its output equals the custom-op output bitwise on
    stub shards; M<=512 falls through to the custom op."""

    class ReconLinear:
        """Stub LinearEXL3: get_weight_tensor returns a known [k, out] fp16."""

        def __init__(self, k, out, seed):
            g = torch.Generator().manual_seed(seed)
            self.w = torch.randn(k, out, dtype=torch.float16, generator=g)
            self.out_features = out

        def get_weight_tensor(self):
            return self.w

        def forward(self, x, params, out_dtype=torch.float16):
            # the custom op computes each shard as one fp16 GEMM
            assert out_dtype == torch.float16
            return torch.nn.functional.linear(x, self.w.t())

    # real TP2 shape: 3 x 4096 EXL3 shards + b(32)/f_a(128)/g_a(128) tail
    sizes = [4096, 4096, 4096, 32, 128, 128]
    n_exl3 = sum(sizes[:3])
    n_local = sum(sizes)
    cfg = _nr_config(mod, layers={
        "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a": {
            "bits": 6, "bf16_shards": [3, 4, 5]}})
    m = mod.Exl3LinearMethod(
        cfg, "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a", 6)
    m.n_shards = len(sizes)
    m.output_sizes = list(sizes)
    m.bf16_shards = [3, 4, 5]
    m.is_row_parallel = False
    m.tp_rank = 0
    m.tp_size = 2
    m.replicated_shards = frozenset((4, 5))
    m.in_per_partition = H

    linears = [ReconLinear(H, 4096, 1), ReconLinear(H, 4096, 2),
               ReconLinear(H, 4096, 3), None, None, None]
    tail = torch.randn(sum(sizes[3:]), H, dtype=torch.bfloat16,
                       generator=torch.Generator().manual_seed(4))

    layer = torch.nn.Module()
    try:
        os.environ["GLM53_KDA_BF16_LARGE_M"] = "1"
        m._retain_bf16_large_m(layer, linears, tail)
    finally:
        os.environ.pop("GLM53_KDA_BF16_LARGE_M", None)
    w16 = layer.glm53_bf16_lm_w16
    assert w16.dtype == torch.float16 and w16.shape == (n_exl3, H)
    assert layer.glm53_bf16_lm_tail is tail
    assert layer.glm53_bf16_lm_min_m == mod.KDA_BF16_LARGE_M_MIN_M
    assert layer.glm53_bf16_lm_n == n_local and layer.glm53_bf16_lm_k == H
    # rows are the recon transposes, kept fp16 (no bf16 cast)
    off = 0
    for i in range(3):
        assert torch.equal(w16[off:off + sizes[i]], linears[i].w.t())
        off += sizes[i]

    # M>512: apply() must equal the custom-op output bitwise. Build the
    # op entry over the same stub shards + tail, then compare.
    mod._DENSE_EXL3_LAYERS.append(
        {"linears": list(linears), "bf16_shards": [3, 4, 5],
         "output_sizes": list(sizes), "bf16_weight": tail})
    layer._exl3_dense_handle = len(mod._DENSE_EXL3_LAYERS) - 1
    x = torch.randn(600, H, dtype=torch.bfloat16)
    y_apply = m.apply(layer, x)
    y_op = mod._dense_exl3_forward_impl(x.reshape(-1, H), layer._exl3_dense_handle)
    assert y_apply.dtype == torch.bfloat16 and y_apply.shape == (600, n_local)
    assert torch.equal(y_apply, y_op), "large-M apply must match the custom op bitwise"
    # and the parts equal the two GEMMs directly
    assert torch.equal(y_apply[:, :n_exl3],
                       torch.nn.functional.linear(x.half(), w16).to(torch.bfloat16))
    assert torch.equal(y_apply[:, n_exl3:],
                       torch.nn.functional.linear(x, tail))

    # M<=512 falls through to the custom op: unregisterable here, so prove
    # the branch was not taken via the stats counter.
    stats0 = mod.kda_large_m_dispatch_stats()["bf16_calls"]
    x_lo = torch.randn(8, H, dtype=torch.bfloat16)
    try:
        m.apply(layer, x_lo)
    except Exception as exc:  # noqa: BLE001  (torch.ops lookup on CPU)
        assert "dense_exl3_forward" in repr(exc), exc
    else:
        raise AssertionError("M<=512 must fall through to the custom op")
    assert mod.kda_large_m_dispatch_stats()["bf16_calls"] == stats0, \
        "M<=512 must not take the large-M branch"
    print("kda bf16-large-m on EXL3 (fp16 rows + bf16 tail, bitwise apply) OK")


def ablit_tests():
    spec = importlib.util.spec_from_file_location("ablit_dense_test", ABLIT_SRC)
    ablit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablit)

    # End to end through apply_ablit: any EXL3 o_proj in range fails loud
    # even before a weight edit could be attempted.
    class Exl3OProj(torch.nn.Module):
        _exl3_linear_n_shards = 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.o_proj = Exl3OProj()
            self.layers["15"] = layer

    model = Model()
    r = torch.ones(4)
    try:
        ablit.apply_ablit(model, r, [15], alpha=3.0, include_mtp=False)
    except ablit.AblitError as exc:
        assert "EXL3-dense" in str(exc), exc
    else:
        raise AssertionError("apply_ablit over an EXL3 o_proj must refuse")
    print("ablit EXL3 o_proj refusal OK")



def forward_op_tests(mod):
    """The custom-op impl on CPU with stub LinearEXL3 handles: bf16 tail
    output must be exactly F.linear(x, w) in x.dtype (never through fp16 —
    a value past fp16 range would become inf), and the no-bf16 paths return
    x.dtype with the EXL3 outputs cast once."""

    class StubLinear:
        out_features = 64

        def forward(self, x, params, out_dtype=torch.float16):
            assert out_dtype == torch.float16
            return torch.zeros(x.shape[0], self.out_features, dtype=torch.float16)

    def entry(linears, output_sizes, bf16_weight):
        mod._DENSE_EXL3_LAYERS.append(
            {"linears": linears, "bf16_shards": [i for i, l in enumerate(linears) if l is None],
             "output_sizes": output_sizes, "bf16_weight": bf16_weight})
        return len(mod._DENSE_EXL3_LAYERS) - 1

    x = torch.randn(4, 128, dtype=torch.bfloat16)
    # bf16 tail whose true outputs exceed fp16 range: a fp16 round-trip
    # would flush them to inf and fail the equality.
    w = torch.zeros(32, 128, dtype=torch.bfloat16)
    w[:, :] = 1e5  # 128 * 1e5 = 1.28e7 > fp16 max 65504
    h = entry([StubLinear(), None], [64, 32], w)
    y = mod._dense_exl3_forward_impl(x, h)
    assert y.dtype == x.dtype, y.dtype
    assert y.shape == (4, 96), y.shape
    tail = torch.nn.functional.linear(x, w)
    assert torch.equal(y[:, 64:], tail), "bf16 tail must be the exact BF16 GEMM"
    assert torch.isfinite(y).all() and float(y[:, 64:].abs().max()) > 65504
    assert float(y[:, :64].abs().max()) == 0.0

    h = entry([StubLinear(), StubLinear()], [64, 64], None)
    y = mod._dense_exl3_forward_impl(x, h)
    assert y.dtype == x.dtype and y.shape == (4, 128)

    h = entry([StubLinear()], [64], None)
    y = mod._dense_exl3_forward_impl(x, h)
    assert y.dtype == x.dtype and y.shape == (4, 64)
    print("dense_exl3_forward impl (bf16 tail exactness, dtypes) OK")


def warmup_tests(mod):
    """The pre-capture autotune warmup must run every unique dense-EXL3 GEMM
    shape at every row bucket derivable from the capture sizes <= 144. The
    autotune hash keys on MIN(roundup_pow2(rows), 16) + dims (image
    coop_autotune.cu / exl3_gemm.cu:gemm_autotune_hash), so bucket coverage
    is exact coverage; dedup across modules sharing a shape is required
    (tune once per shape x bucket)."""

    calls = []

    class T:  # bare tensor carrier for .device
        device = torch.device("cpu")

    class StubLin:
        """mcg/mul1 as bools, exactly like the image's LinearEXL3."""

        def __init__(self, in_f, out_f, K, cb):
            self.in_features, self.out_features, self.K = in_f, out_f, K
            self.mcg = cb == "mcg"
            self.mul1 = cb == "mul1"
            self.trellis = T()

        def forward(self, x, params, out_dtype=torch.float16):
            assert out_dtype == torch.float16
            calls.append((self.in_features, self.out_features, self.K,
                          "mcg" if self.mcg else "mul1", x.shape[0]))
            return torch.zeros(x.shape[0], self.out_features, dtype=torch.float16)

    a1 = StubLin(4096, 8192, 6, "mul1")
    a2 = StubLin(4096, 8192, 6, "mul1")  # same shape as a1 -> dedup
    b = StubLin(1536, 8192, 6, "mcg")    # distinct shape
    mod._DENSE_EXL3_LAYERS.clear()
    mod._DENSE_EXL3_LAYERS.append(
        {"linears": [a1, a2, b], "bf16_shards": [], "output_sizes": [8192, 8192, 8192],
         "bf16_weight": None})

    orig_avail = torch.cuda.is_available
    orig_sync = torch.cuda.synchronize
    torch.cuda.is_available = lambda: True
    torch.cuda.synchronize = lambda *a, **k: None
    try:
        # capture sizes 1,2,4,8 bucket to themselves; 24, 96 bucket to 16;
        # 512 exceeds the 144 reconstruct threshold and must NOT be warmed.
        n = mod._dense_exl3_warmup_autotune([1, 2, 4, 8, 24, 96, 512])
    finally:
        torch.cuda.is_available = orig_avail
        torch.cuda.synchronize = orig_sync

    buckets = {1, 2, 4, 8, 16}
    assert n == 2 * len(buckets), n  # two unique shapes x five buckets
    per_shape = {}
    for in_f, out_f, K, cb, rows in calls:
        per_shape.setdefault((in_f, out_f, K, cb), []).append(rows)
        assert rows in buckets, rows  # nothing > 144, nothing odd
    assert set(per_shape) == {(4096, 8192, 6, "mul1"), (1536, 8192, 6, "mcg")}
    for shape, rows in per_shape.items():
        assert sorted(rows) == sorted(buckets), (shape, rows)

    # unreadable capture sizes -> all five buckets warmed (fail-safe)
    calls.clear()
    torch.cuda.is_available = lambda: True
    torch.cuda.synchronize = lambda *a, **k: None
    try:
        n = mod._dense_exl3_warmup_autotune(None)
    finally:
        torch.cuda.is_available = orig_avail
        torch.cuda.synchronize = orig_sync
    assert n == 2 * 5 and {r for *_, r in calls} == buckets
    print("coop-autotune warmup (shape dedup, bucket coverage) OK")


def main() -> int:
    stubs = _install_vllm_stubs()
    mod = _load_exl3()
    config_tests(mod)
    geometry_tests(mod, stubs)
    forward_op_tests(mod)
    dispatch_tests(mod, stubs)
    large_m_tests(mod)
    warmup_tests(mod)
    ablit_tests()
    print("dense-exl3 overlay CPU checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
