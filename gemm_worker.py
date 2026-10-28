"""8192^3 torch.mm worker. Reads one JSON command per stdin line, answers with a
'@@RESULT {json}' line on stdout."""
import hashlib
import json
import sys
import time

import torch

N = 8192
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32}
# ~6 s per run at stock, so the 0.5 s sampler gets a dozen samples
ITERS = {"bf16": 400, "fp16": 400, "tf32": 200}

# Zagitta/gpu-unlock reports measure with FP32 reductions (upstream #165, #158, #71)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_accumulation = False

_operands = {}


def operands(kind):
    if kind not in _operands:
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(N, N, device="cuda", dtype=DTYPES[kind], generator=g)
        b = torch.randn(N, N, device="cuda", dtype=DTYPES[kind], generator=g)
        _operands.clear()
        _operands[kind] = (a, b)
    return _operands[kind]


def run(kind, iters=None):
    torch.backends.cuda.matmul.allow_tf32 = kind == "tf32"
    a, b = operands(kind)
    iters = iters or ITERS[kind]
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3
    digest = hashlib.sha256(c.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return {"seconds": seconds, "tflops": 2 * N**3 * iters / seconds / 1e12, "sha256": digest}


def heat(seconds):
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = operands("bf16")
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for _ in range(50):
            torch.mm(a, b)
        torch.cuda.synchronize()
    return {"seconds": time.monotonic() - t0}


def main():
    print("@@RESULT " + json.dumps({"ready": True, "torch": torch.__version__}), flush=True)
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd["op"] == "gemm":
            res = run(cmd["kind"], cmd.get("iters"))
        elif cmd["op"] == "heat":
            res = heat(cmd["seconds"])
        else:
            res = {"error": f"unknown op {cmd['op']}"}
        print("@@RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
