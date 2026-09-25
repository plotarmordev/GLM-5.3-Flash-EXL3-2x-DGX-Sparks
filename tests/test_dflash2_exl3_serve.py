#!/usr/bin/env python3
"""CPU checks for the EXL3 DFlash2 draft serving path (arm ED):

- overlay/patch_dflash2_exl3.py (the GLM53_OVERLAY_ORDER runtime overlay)
  applies its two [dense-exl3-dflash2] anchors to the hash-pinned image
  qwen3_dflash.py and installs the mounted qwen3_dflash2.py over the baked
  copy — idempotent, fail-closed on drift, output compiles.
- The fused context-KV block is byte-identical for a BF16 draft and uses the
  EXL3 staging weight (k/v rows only) whole when q is EXL3.
- overlay/qwen3_dflash2.py threads quant_config into DFlashGroupedConv (both
  call sites) and keeps CandidateSelector.hidden_projection unquantized.
- _log_draft_exl3 emits the D6 boot line (count + staged bytes) for an EXL3
  draft, runs the declared-vs-built check, and is a no-op for a BF16 draft.

Runs with real torch and stubbed vllm modules; never imports vllm.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH_SRC = ROOT / "overlay" / "patch_dflash2_exl3.py"
DFLASH2_SRC = ROOT / "overlay" / "qwen3_dflash2.py"
FIXTURE = HERE / "fixtures" / "qwen3_dflash-927d6521.py.txt"
# sha256 of the image's qwen3_dflash.py (sha256:927d6521…, post-overlay state
# as patch_dflash2.py leaves it). Drift here means the anchors must be
# re-derived against the new image.
FIXTURE_SHA256 = "40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745"


def _load_patch():
    spec = importlib.util.spec_from_file_location("patch_dflash2_test", PATCH_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_patcher(site: Path, opt: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GLM53_SITE"] = str(site)
    env["GLM53_OPT"] = str(opt)
    return subprocess.run(
        [sys.executable, str(PATCH_SRC)], env=env, capture_output=True, text=True
    )


def _stage_runtime(tmp: Path, qwen_bytes: bytes) -> tuple[Path, Path, Path, Path]:
    """A scratch container filesystem: baked image files under `site`, the
    host-mounted overlay dir under `opt` (as the docker -v mounts land)."""
    site = tmp / "site"
    models = site / "model_executor" / "models"
    models.mkdir(parents=True)
    qwen = models / "qwen3_dflash.py"
    qwen.write_bytes(qwen_bytes)
    baked = models / "qwen3_dflash2.py"
    baked.write_text("# stale baked copy from the image build\n")
    opt = tmp / "opt"
    opt.mkdir()
    (opt / "qwen3_dflash2.py").write_bytes(DFLASH2_SRC.read_bytes())
    return site, opt, qwen, baked


def test_runtime_patcher() -> None:
    mod = _load_patch()
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, f"image fixture drifted: {digest}"
    # Expected post-patch bytes, derived from the same constants.
    expected = mod._apply(
        FIXTURE.read_text(), mod.QUANT_PREFIX_SHIFT_OLD, mod.QUANT_PREFIX_SHIFT_NEW, "shift"
    )
    expected = mod._apply(expected, mod.FUSED_KV_OLD, mod.FUSED_KV_NEW, "kv")
    compile(expected, "expected", "exec")

    with tempfile.TemporaryDirectory() as t:
        site, opt, qwen, baked = _stage_runtime(Path(t), FIXTURE.read_bytes())
        # A baked __pycache__ from the image build must be cleared on patch.
        cache = qwen.parent / "__pycache__"
        cache.mkdir()
        (cache / "qwen3_dflash.cpython-312.pyc").write_bytes(b"stale")

        r = _run_patcher(site, opt)
        assert r.returncode == 0, r.stderr
        assert "patched qwen3_dflash.py" in r.stdout, r.stdout
        assert "installed qwen3_dflash2.py" in r.stdout, r.stdout
        assert qwen.read_text() == expected
        assert baked.read_bytes() == DFLASH2_SRC.read_bytes()
        assert not list(cache.glob("qwen3_dflash*.pyc"))  # stale pyc cleared

        # Re-run (every container start; fresh builds already carry it):
        # byte-identical no-op.
        r = _run_patcher(site, opt)
        assert r.returncode == 0, r.stderr
        assert "already present" in r.stdout, r.stdout
        assert qwen.read_text() == expected
        assert baked.read_bytes() == DFLASH2_SRC.read_bytes()

    # Drift: neither anchor state matches -> fail closed, boot dies loudly.
    with tempfile.TemporaryDirectory() as t:
        drifted = FIXTURE.read_text().replace(mod.FUSED_KV_OLD, "")
        assert drifted != FIXTURE.read_text()
        site, opt, qwen, _ = _stage_runtime(Path(t), drifted.encode())
        r = _run_patcher(site, opt)
        assert r.returncode != 0
        assert "fused context-KV" in r.stderr, r.stderr
    print("runtime patcher (pinned image, install, idempotent, drift-refusal) OK")


def test_fused_kv_block() -> None:
    mod = _load_patch()
    """Exec the exact block the patch writes into _build_context_kv_buffers."""

    class Attn:
        def __init__(self, q_size, kv_size, weight):
            self.q_size = q_size
            self.kv_size = kv_size
            self.qkv_proj = types.SimpleNamespace(weight=weight)

    def run_block(layers_attn):
        ns = {"torch": torch, "layers_attn": layers_attn, "self": types.SimpleNamespace()}
        exec(textwrap.dedent(mod.FUSED_KV_NEW), ns)
        return ns["kv_weights"]

    g = torch.Generator().manual_seed(7)
    q, kv, h = 2048, 512, 4096  # TP2-local draft geometry
    fulls = [
        torch.randn(q + 2 * kv, h, dtype=torch.bfloat16, generator=g)
        for _ in range(5)
    ]
    kv_rows = [torch.randn(2 * kv, h, dtype=torch.bfloat16, generator=g) for _ in range(5)]

    # BF16 draft: the block returns exactly the legacy expression's rows.
    got = run_block([Attn(q, kv, w) for w in fulls])
    legacy = [w[q:] for w in fulls]
    assert len(got) == 5
    for a, b in zip(got, legacy):
        assert a.data_ptr() == b.data_ptr()  # same view, no copy
        assert torch.equal(a, b)
    assert torch.equal(torch.cat(got, dim=0), torch.cat(legacy, dim=0))

    # EXL3 draft (q EXL3, k/v bf16): the staging weight IS the KV block and
    # must be used whole — no q rows to slice off.
    got = run_block([Attn(q, kv, w) for w in kv_rows])
    for a, b in zip(got, kv_rows):
        assert a.data_ptr() == b.data_ptr()
    assert torch.equal(torch.cat(got, dim=0), torch.cat(kv_rows, dim=0))

    # any other shape refuses loudly instead of building a wrong fused weight
    bad = [Attn(q, kv, torch.randn(kv, h, dtype=torch.bfloat16, generator=g))]
    try:
        run_block(bad)
    except AssertionError as exc:
        assert "neither the full QKV" in str(exc), exc
    else:
        raise AssertionError("unexpected qkv_proj.weight shape must refuse")
    print("fused context-KV block (bf16 identity, exl3 staging, refusal) OK")


def _load_dflash2_overlay():
    """Load overlay/qwen3_dflash2.py with stub vllm + stub sibling modules.

    ReplicatedLinear records its kwargs; the sibling qwen3_dflash stub
    provides plain base classes; maybe_prefix is the real contract
    (dot-join)."""
    stubs = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    import contextlib

    mod("vllm.compilation.backends",
        set_model_tag=lambda tag: contextlib.nullcontext())
    mod("vllm.compilation.decorators", support_torch_compile=lambda cls: cls)

    class VllmConfig:
        pass

    class CacheConfig:
        pass

    mod("vllm.config", VllmConfig=VllmConfig, CacheConfig=CacheConfig)
    mod("vllm.logger", init_logger=lambda name: logging.getLogger(name))

    class ReplicatedLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, bias, params_dtype,
                     quant_config, prefix, return_bias):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.quant_config_arg = quant_config
            self.prefix = prefix

    mod("vllm.model_executor.layers.linear", ReplicatedLinear=ReplicatedLinear)

    class LogitsProcessor:
        def __init__(self, *a, **k):
            pass

    mod("vllm.model_executor.layers.logits_processor",
        LogitsProcessor=LogitsProcessor)

    class QuantizationConfig:
        pass

    mod("vllm.model_executor.layers.quantization.base_config",
        QuantizationConfig=QuantizationConfig)

    # package + sibling stubs for the relative imports
    pkg = types.ModuleType("dflash2_overlay_test")
    pkg.__path__ = []
    sys.modules["dflash2_overlay_test"] = pkg

    class DFlashQwen3DecoderLayer(torch.nn.Module):
        def __init__(self, vllm_config, **kw):
            super().__init__()
            self.base_quant_config = kw.get("quant_config")

    class DFlashQwen3Model(torch.nn.Module):
        pass

    class DFlashQwen3ForCausalLM(torch.nn.Module):
        pass

    mod("dflash2_overlay_test.qwen3_dflash",
        DFlashQwen3DecoderLayer=DFlashQwen3DecoderLayer,
        DFlashQwen3Model=DFlashQwen3Model,
        DFlashQwen3ForCausalLM=DFlashQwen3ForCausalLM)
    mod("dflash2_overlay_test.utils",
        maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name)

    spec = importlib.util.spec_from_file_location(
        "dflash2_overlay_test.qwen3_dflash2", DFLASH2_SRC)
    overlay = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = overlay
    spec.loader.exec_module(overlay)
    stubs["ReplicatedLinear"] = ReplicatedLinear
    return overlay, stubs


def test_quant_config_threading() -> None:
    overlay, _ = _load_dflash2_overlay()
    sentinel = object()

    conv = overlay.DFlashGroupedConv(
        hidden_size=4096, taps=4, group_size=16, block_size=8,
        params_dtype=torch.bfloat16, prefix="model.layers.45.attention_conv",
        quant_config=sentinel)
    assert conv.kernel_projection.quant_config_arg is sentinel
    assert conv.kernel_projection.prefix == "model.layers.45.attention_conv.kernel_projection"

    # default stays None: the BF16 draft builds byte-identical modules
    conv = overlay.DFlashGroupedConv(
        hidden_size=4096, taps=4, group_size=16, block_size=8,
        params_dtype=torch.bfloat16, prefix="x")
    assert conv.kernel_projection.quant_config_arg is None

    selector = overlay.CandidateSelector(
        hidden_size=4096, vocab_size=154880, rank=256, top_k=8,
        params_dtype=torch.bfloat16, prefix="model.candidate_selector")
    # hidden_projection is deliberately BF16 (selector steers acceptance)
    assert selector.hidden_projection.quant_config_arg is None

    # both decoder-layer call sites thread the layer's quant_config
    vllm_config = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(num_speculative_tokens=7),
        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
    )
    config = types.SimpleNamespace(
        hidden_size=4096,
        dflash_config={"conv_kernel_size": 4, "conv_group_size": 16},
    )
    layer = overlay.DFlash2Qwen3DecoderLayer(
        vllm_config, config=config, layer_idx=0, quant_config=sentinel,
        prefix="model.layers.45")
    assert layer.base_quant_config is sentinel  # super() got it (attn/mlp)
    assert layer.attention_conv.kernel_projection.quant_config_arg is sentinel
    assert layer.mlp_conv.kernel_projection.quant_config_arg is sentinel
    print("quant_config threading (convs yes, selector no) OK")


def test_draft_boot_log() -> None:
    overlay, _ = _load_dflash2_overlay()
    records = []

    class _Cap(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    log = logging.getLogger("dflash2_overlay_test.qwen3_dflash2")
    cap = _Cap()
    log.addHandler(cap)
    log.setLevel(logging.INFO)

    class FakeLinear(torch.nn.Module):
        def __init__(self, with_marker):
            super().__init__()
            if with_marker:
                self._exl3_linear_n_shards = 1
                self.trellis = torch.zeros(4, 4, 80, dtype=torch.int16)
                self.suh = torch.zeros(1, 64, dtype=torch.float16)
                self.svh = torch.zeros(64, dtype=torch.float16)
                self.mcg = torch.zeros(1, 1, dtype=torch.int32)
                self.mul1 = torch.zeros(1, 1, dtype=torch.int32)
                self.weight = torch.zeros(32, 64, dtype=torch.bfloat16)

    class FakeQuantConfig:
        def __init__(self, declared):
            self.non_routed_exl3 = {"layers": declared}
            self.assert_calls = 0

        def _assert_non_routed_built(self):
            self.assert_calls += 1

    class FakeModel(torch.nn.Module):
        def __init__(self, quant_config, n_exl3):
            super().__init__()
            self.quant_config = quant_config
            self.linears = torch.nn.ModuleList(
                [FakeLinear(True) for _ in range(n_exl3)]
                + [FakeLinear(False)])  # unquantized module ignored

    try:
        # BF16 draft: no declarations -> no log, no check, no exception
        qc = FakeQuantConfig({})
        overlay._log_draft_exl3(FakeModel(qc, 0))
        assert qc.assert_calls == 0 and not records

        # EXL3 draft: check runs once, one line with count + bytes
        declared = {f"model.layers.{45 + i}.self_attn.qkv_proj": {"bits": 5}
                    for i in range(5)}
        qc = FakeQuantConfig(declared)
        overlay._log_draft_exl3(FakeModel(qc, 31))
        assert qc.assert_calls == 1
        assert len(records) == 1, records
        line = records[0]
        assert line.startswith("[dense-exl3] draft: 31 EXL3 modules loaded ("), line
        assert "MiB staged" in line and line.endswith(")"), line
        # exact staged bytes: 31 x (4*4*80*2 + 64*2 + 64*2 + 4 + 4 + 32*64*2)
        per = 4 * 4 * 80 * 2 + 64 * 2 + 64 * 2 + 4 + 4 + 32 * 64 * 2
        assert f"{31 * per / 2**20:.1f} MiB" in line, line
    finally:
        log.removeHandler(cap)
    print("draft EXL3 boot line (count + bytes, bf16 no-op) OK")

def test_launcher_wiring() -> None:
    """The runtime-overlay plumbing: Dockerfile build slot, GLM53_OVERLAY_ORDER,
    both ranks' mounts + worker scp, and the host artifact guard."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    copy_line = "COPY overlay/patch_dflash2_exl3.py /opt/glm53/patch_dflash2_exl3.py"
    run_line = "RUN python3 /opt/glm53/patch_dflash2_exl3.py"
    assert copy_line in dockerfile and run_line in dockerfile
    # fresh build applies it right after the base DFlash2 overlay
    assert dockerfile.index(run_line) > dockerfile.index(
        "RUN python3 /opt/glm53/patch_dflash2.py"
    )

    start = (ROOT / "start.sh").read_text()
    order = start[start.index("GLM53_OVERLAY_ORDER=("):]
    order = order[: order.index(")\n")]
    assert "patch_dflash2_exl3.py" in order
    # head mount, worker scp + mount, artifact guard, host override knob
    assert '-v "$DFLASH2_EXL3_PATCH_HOST:/opt/glm53/patch_dflash2_exl3.py:ro"' in start
    assert '-v "$DFLASH2_MODEL_OVERLAY_HOST:/opt/glm53/qwen3_dflash2.py:ro"' in start
    assert '"${WORKER_SSH}:/tmp/patch_dflash2_exl3.py"' in start
    assert '"${WORKER_SSH}:/tmp/glm53-qwen3_dflash2.py"' in start
    assert "-v '/tmp/patch_dflash2_exl3.py:/opt/glm53/patch_dflash2_exl3.py:ro'" in start
    assert "-v '/tmp/glm53-qwen3_dflash2.py:/opt/glm53/qwen3_dflash2.py:ro'" in start
    assert '"$DFLASH2_EXL3_PATCH_HOST|[dense-exl3-dflash2]|$main_guard"' in start
    assert '"$DFLASH2_MODEL_OVERLAY_HOST|DFlash2Qwen3ForCausalLM|' in start
    assert 'DFLASH2_EXL3_PATCH_HOST="${DFLASH2_EXL3_PATCH_HOST:-' in start
    print("launcher wiring (order, mounts, scp, artifact guard, Dockerfile) OK")


def main() -> int:
    test_runtime_patcher()
    test_fused_kv_block()
    test_quant_config_threading()
    test_draft_boot_log()
    test_launcher_wiring()
    print("dflash2 EXL3 serving CPU checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
