#!/usr/bin/env python3
"""GPU Unlock benchmark runner: unlock x undervolt matrix on one NVIDIA GPU.

  bench.py session --session N --unlock on|off   one session (1, 2: ABBA; 3: stock cell only)
  bench.py stock-ref                             reference hashes at unlock off, undervolt off
  bench.py soak --unlock on|off                  2 h of SD 1.5 + Z-Image checked against the reference

Run through ./run.sh. Environment:
  BENCH_DIR       state dir: vllm-venv/, models/Qwen3-8B/, zimage.sha256, sd15.sha256
  COMFY_DIR       ComfyUI install (with .venv/ and ComfyUI/)
  COMFY_URL       ComfyUI API (default http://127.0.0.1:8188)
  UNDERVOLT_UNIT  systemd unit that applies the undervolt on start and restores stock on stop (default gpu-tune)
  WATCHDOG_UNIT   optional unit started with it that falls back on an Xid (default gpu-tune-watchdog)
  TRIPPED_MARKER  file the watchdog writes on an Xid (default /run/gpu-tune/tripped)
  COMFY_UNIT      ComfyUI's systemd unit (default comfyui)
"""
import argparse
import csv
import datetime
import glob
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

import pynvml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ["BENCH_DIR"]
QWEN_DIR = f"{STATE}/models/Qwen3-8B"
QWEN_REPO, QWEN_REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
ZIMAGE_REPO, ZIMAGE_REV = "Comfy-Org/Z-Image-Turbo", "6fc90a3b1b653e935a0d175e260736de25b84df5"
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ["COMFY_DIR"]
COMFY_UNIT = os.environ.get("COMFY_UNIT", "comfyui")
UV_UNIT = os.environ.get("UNDERVOLT_UNIT", "gpu-tune")
UV_UNITS = [UV_UNIT] + [u for u in [os.environ.get("WATCHDOG_UNIT", "gpu-tune-watchdog")] if u]
MARKER = os.environ.get("TRIPPED_MARKER", "/run/gpu-tune/tripped")
STOCK_POWER_LIMIT_W = 350.0
SEED = 42
DROP_BELOW_W = 60.0

RUN_FIELDS = ["session", "boot_id", "unlock", "undervolt", "workload", "run", "seconds",
              "throughput", "avg_w", "joules", "avg_sm_mhz", "peak_c", "peak_fan",
              "output_sha256", "pass"]
SMI_FIELDS = ["power.draw", "clocks.sm", "temperature.gpu", "fan.speed", "power.limit", "clocks.mem", "pstate"]
SAMPLE_FIELDS = ["session", "unlock", "undervolt", "workload", "run", "t_unix", "power_w", "sm_mhz",
                 "temp_c", "fan_pct", "power_limit_w", "mem_mhz", "pstate"]

GEMM = ["gemm_bf16", "gemm_fp16", "gemm_tf32"]
LLM = ["vllm_prefill", "vllm_decode_b1", "vllm_decode_b32", "llm_tokens"]
IMAGES = ["sd15", "zimage"]
HASHED = GEMM + ["llm_tokens"] + IMAGES


class Abort(Exception):
    pass


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


class Sampler:
    """nvidia-smi every 0.5 s for the duration of one run."""

    def __init__(self):
        self.rows = []

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            vals = [v.strip() for v in line.split(",")]
            if len(vals) == len(SMI_FIELDS):
                self.rows.append([time.time()] + vals)

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        self.thread.join()


def num(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


class Worker:
    """Long-lived subprocess speaking JSON lines; '@@RESULT ' marks replies."""

    def __init__(self, script, *args):
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, script), *args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.hello = self._reply()

    def _reply(self):
        for line in self.proc.stdout:
            if line.startswith("@@RESULT "):
                res = json.loads(line[len("@@RESULT "):])
                if "error" in res:
                    raise Abort(res["error"])
                return res
            sys.stderr.write(line)
        raise Abort(f"worker exited with {self.proc.wait()}")

    def call(self, **cmd):
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return self._reply()

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=120)


# ---------------------------------------------------------------- ComfyUI

def http_json(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(COMFY + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def comfy_up():
    sh("systemctl", "start", COMFY_UNIT)
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            http_json("/system_stats", timeout=5)
            return
        except OSError:
            time.sleep(2)
    raise Abort("ComfyUI did not come up within 15 min")


def comfy_down():
    sh("systemctl", "stop", COMFY_UNIT)


def comfy_wait_idle():
    """True if someone else's prompt had to finish first: their models may have
    evicted ours, so that run's timing (and, under VRAM pressure, its pixels) is suspect."""
    waited = False
    while True:
        q = http_json("/queue")
        if not q["queue_running"] and not q["queue_pending"]:
            return waited
        if not waited:
            log("ComfyUI queue busy with someone else's prompt, waiting")
        waited = True
        time.sleep(5)


def comfy_free():
    # start each image workload from an empty VRAM, not whatever the last user left loaded
    req = urllib.request.Request(COMFY + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).close()
    time.sleep(3)


def sd15_workflow(nonce):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"}},
        # the nonce is ignored by the node but changes the cache key, so the
        # sampler and decoder rerun while loaders and text encoding stay cached
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 4, "_bench_nonce": nonce}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "a lighthouse on a rocky coast at sunset, detailed oil painting"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "blurry, low quality"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": 30, "cfg": 8.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                                                   "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
    }, "9", 4


def zimage_workflow(nonce):
    # settings from ComfyUI's image_z_image_turbo template
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up."}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1, "_bench_nonce": nonce}},
        "8": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0,
                                                   "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    }, "10", 1


def comfy_generate(workflow_fn):
    interfered = comfy_wait_idle()
    wf, out_node, batch = workflow_fn(uuid.uuid4().hex)
    pid = http_json("/prompt", {"prompt": wf, "client_id": "gpu-unlock-bench"})["prompt_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        hist = http_json(f"/history/{pid}")
        if pid in hist and hist[pid]["status"].get("completed") is not None:
            break
        time.sleep(0.25)
    else:
        raise Abort(f"ComfyUI prompt {pid} timed out")
    h = hist[pid]
    if h["status"]["status_str"] != "success":
        raise Abort(f"ComfyUI prompt failed: {json.dumps(h['status'])[:2000]}")
    ts = {m[0]: m[1]["timestamp"] for m in h["status"]["messages"]}
    seconds = (ts["execution_success"] - ts["execution_start"]) / 1e3
    digest = hashlib.sha256()
    for img in h["outputs"][out_node]["images"]:
        q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
        with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=60) as r:
            im = Image.open(io.BytesIO(r.read()))
            im.load()
        # pixels, not PNG bytes, so metadata can never cause a mismatch
        digest.update(f"{im.mode}{im.size}".encode())
        digest.update(im.tobytes())
    return {"seconds": seconds, "throughput": batch / seconds, "sha256": digest.hexdigest(), "interfered": interfered}


# ---------------------------------------------------------------- environment

def read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def dist_version(site, name):
    hits = glob.glob(f"{site}/{name}-*.dist-info")
    return os.path.basename(hits[0])[len(name) + 1:-len(".dist-info")] if hits else None


def collect_env(args):
    smi = sh("nvidia-smi", "--query-gpu=name,pci.bus_id,driver_version,vbios_version,power.default_limit,power.limit",
             "--format=csv,noheader").split(", ")
    vllm_site = glob.glob(f"{STATE}/vllm-venv/lib/python3*/site-packages")[0]
    comfy_site = glob.glob(f"{COMFY_DIR}/.venv/lib/python3*/site-packages")[0]
    return {
        "time": datetime.datetime.now().astimezone().isoformat(),
        "args": vars(args),
        "boot_id": read("/proc/sys/kernel/random/boot_id"),
        "gpu": dict(zip(["name", "pci", "driver", "vbios", "default_power_limit", "power_limit"], smi)),
        "kernel": os.uname().release,
        "cmdline": read("/proc/cmdline"),
        "nixos_version": sh("nixos-version", check=False),
        "nixos_generation": os.readlink("/nix/var/nix/profiles/system"),
        "current_system": os.readlink("/run/current-system"),
        "gpu_unlock": {
            "lsmod": [l for l in sh("lsmod").splitlines() if "unlock" in l],
            "modinfo": sh("modinfo", "gpu_unlock", check=False),
            "kernel_log": [l for l in sh("journalctl", "-k", "-b", "-o", "short-monotonic", "--no-pager", check=False).splitlines()
                           if "unlock" in l.lower()][-60:],
        },
        "gpu_tune": {"active": sh("systemctl", "is-active", UV_UNIT, check=False), "tripped_marker": os.path.exists(MARKER)},
        "versions": {
            "bench_commit": sh("git", "-C", HERE, "rev-parse", "HEAD", check=False),
            "bench_dirty": bool(sh("git", "-C", HERE, "status", "--porcelain", "--", ":!results", check=False)),
            "vllm_venv": {n: dist_version(vllm_site, n) for n in ("torch", "vllm", "flashinfer_python", "triton")},
            "comfyui_venv": {n: dist_version(comfy_site, n) for n in ("torch", "torchvision", "triton", "comfyui_frontend_package")},
            "comfyui_commit": sh("git", "-c", "safe.directory=*", "-C", f"{COMFY_DIR}/ComfyUI", "rev-parse", "HEAD", check=False),
            "python": sys.version,
        },
        "models": {
            "qwen3_8b": {"repo": QWEN_REPO, "revision": QWEN_REV, "path": QWEN_DIR, "sha256sums": read(f"{QWEN_DIR}/SHA256SUMS")},
            "z_image_turbo": {"repo": ZIMAGE_REPO, "revision": ZIMAGE_REV, "sha256sums": read(f"{STATE}/zimage.sha256")},
            "sd15": {"file": "v1-5-pruned-emaonly-fp16.safetensors", "sha256sums": read(f"{STATE}/sd15.sha256")},
        },
        "bench_env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "TRITON_", "CUDA_", "HF_")) or k in ("LD_LIBRARY_PATH", "CC")},
    }


def unlock_loaded():
    return any("gpu_unlock" in l for l in sh("lsmod").splitlines())


# ---------------------------------------------------------------- runner

class Bench:
    def __init__(self, args, label):
        self.args = args
        self.session = label
        self.unlock = args.unlock
        self.uv = None
        self.boot_id = read("/proc/sys/kernel/random/boot_id")
        self.started = time.time()
        self.out = os.path.join(HERE, "results", label)
        if os.path.exists(os.path.join(self.out, "runs.csv")) and not args.force:
            raise SystemExit(f"{self.out} already has results; pick another --label or pass --force")
        os.makedirs(self.out, exist_ok=True)
        self.runs = self._csv("runs.csv", RUN_FIELDS)
        self.samples = self._csv("samples.csv", SAMPLE_FIELDS)
        self.ref = {}
        if args.reference and os.path.exists(args.reference):
            with open(args.reference) as f:
                self.ref = json.load(f)
        pynvml.nvmlInit()
        self.nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.gemm = self.llm = None
        self.failures = []

    def _csv(self, name, fields):
        path = os.path.join(self.out, name)
        new = not os.path.exists(path)
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        return f, w

    def _row(self, which, row):
        f, w = which
        w.writerow(row)
        f.flush()

    # -- safety

    def xids(self):
        out = sh("journalctl", "-k", "--since", f"@{int(self.started)}", "-o", "cat", "--no-pager", check=False)
        return [l for l in out.splitlines() if "Xid (" in l]

    def guard(self):
        if os.path.exists(MARKER):
            raise Abort(f"{MARKER} exists ({read(MARKER)}); not clearing it, stop and investigate")
        if x := self.xids():
            raise Abort(f"Xid during the run: {x[0]}")
        pl = num(sh("nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"))
        if abs(pl - STOCK_POWER_LIMIT_W) > 0.5:
            raise Abort(f"power limit is {pl} W, expected {STOCK_POWER_LIMIT_W} W")

    def set_uv(self, uv):
        if uv == "on":
            sh("systemctl", "start", *UV_UNITS)
        else:
            # the unit's ExecStop must restore offset 0, -rgc and the default power limit
            sh("systemctl", "stop", *reversed(UV_UNITS))
        time.sleep(2)
        offset = pynvml.nvmlDeviceGetGpcClkVfOffset(self.nvml)
        active = sh("systemctl", "is-active", UV_UNIT, check=False)
        if (uv == "on") != (active == "active") or (uv == "on") != (offset != 0):
            raise Abort(f"undervolt {uv} not applied: {UV_UNIT} {active}, GPC offset {offset} MHz")
        self.guard()
        self.uv = uv
        log(f"undervolt {uv} ({UV_UNIT} {active}, offset {offset} MHz, 350 W, no marker)")

    def restore(self):
        log("restoring: undervolt started, ComfyUI running")
        subprocess.run(["systemctl", "start", *UV_UNITS])
        for w in (self.gemm, self.llm):
            if w and w.proc.poll() is None:
                w.proc.kill()
        try:
            comfy_up()
        except Abort as e:
            log(f"WARNING: {e}")
        if os.path.exists(MARKER):
            log(f"WARNING: {MARKER} present: {read(MARKER)}")

    # -- measurement

    def measure(self, workload, run, fn, timed=True):
        with Sampler() as s:
            res = fn()
        for r in s.rows:
            self._row(self.samples, [self.session, self.unlock, self.uv, workload, run if timed else 0] + r)
        kept = [r for r in s.rows if num(r[1]) >= DROP_BELOW_W]
        avg_w = statistics.fmean(num(r[1]) for r in kept) if kept else float("nan")
        avg_mhz = statistics.fmean(num(r[2]) for r in kept) if kept else float("nan")
        peak_c = max((num(r[3]) for r in s.rows), default=float("nan"))
        peak_fan = max((num(r[4]) for r in s.rows), default=float("nan"))
        verdict = self.verdict(workload, res) if timed else ""
        if "tokens" in res:
            with open(os.path.join(self.out, "llm_tokens.jsonl"), "a") as f:
                f.write(json.dumps({"undervolt": self.uv, "run": run if timed else 0, "sha256": res["sha256"],
                                    "pass": verdict, "tokens": res["tokens"]}) + "\n")
        if res.get("interfered"):
            log(f"WARNING: another ComfyUI user ran before {workload} run {run}; logged to interference.txt")
            with open(os.path.join(self.out, "interference.txt"), "a") as f:
                f.write(f"{time.time():.0f} {workload} undervolt={self.uv} run={run if timed else 'warmup'}\n")
        if timed:
            self._row(self.runs, [self.session, self.boot_id, self.unlock, self.uv, workload, run,
                                  f"{res['seconds']:.4f}", f"{res['throughput']:.4f}", f"{avg_w:.1f}",
                                  f"{avg_w * res['seconds']:.1f}", f"{avg_mhz:.0f}", f"{peak_c:.0f}",
                                  f"{peak_fan:.0f}", res.get("sha256", ""), verdict])
            if verdict == "0":
                self.failures.append((workload, self.uv, run))
        log(f"{workload:16s} uv={self.uv:3s} run={'warmup' if not timed else run:<6} "
            f"{res['seconds']:8.2f} s  thr={res['throughput']:10.2f}  {avg_w:5.0f} W  {avg_mhz:4.0f} MHz  "
            f"{peak_c:3.0f} C  fan {peak_fan:3.0f}%  pass={verdict or '-'}")
        self.guard()
        return res

    def verdict(self, workload, res):
        ref = self.ref.get(workload)
        if not ref or "sha256" not in res:
            return ""
        if res["sha256"] == ref["sha256"]:
            return "1"
        if workload == "llm_tokens":
            diff = [i for i, (a, b) in enumerate(zip(res["tokens"], ref["tokens"])) if a != b]
            log(f"llm_tokens differ from stock in prompts {diff}")
            # stock vs stock already differed, so a mismatch is recorded, not failed
            if not ref.get("deterministic", False):
                return "nd"
        return "0"

    # -- workloads

    def run_fn(self, workload):
        a = self.args
        if workload in GEMM:
            kind = workload.split("_")[1]

            def f():
                r = self.gemm.call(op="gemm", kind=kind)
                return {"seconds": r["seconds"], "throughput": r["tflops"], "sha256": r["sha256"]}
            return f
        if workload in ("vllm_prefill", "vllm_decode_b1", "vllm_decode_b32"):
            spec = {"vllm_prefill": (4096, 1, a.prefill_prompts), "vllm_decode_b1": (128, 512, 1),
                    "vllm_decode_b32": (128, 512, 32)}[workload]

            def f():
                r = self.llm.call(op="random", input_len=spec[0], output_len=spec[1], num_prompts=spec[2])
                toks = r["prompt_tokens"] if workload == "vllm_prefill" else r["output_tokens"]
                return {"seconds": r["seconds"], "throughput": toks / r["seconds"]}
            return f
        if workload == "llm_tokens":
            def f():
                r = self.llm.call(op="tokens", max_tokens=128)
                return {"seconds": r["seconds"], "throughput": r["output_tokens"] / r["seconds"],
                        "sha256": r["sha256"], "tokens": r["tokens"]}
            return f
        return lambda: comfy_generate(sd15_workflow if workload == "sd15" else zimage_workflow)

    def blocks(self, workload, cells):
        """ABBA (or the given cell order) for one workload: 1 warm-up + N timed per block."""
        timed = self.args.token_runs if workload == "llm_tokens" else self.args.timed
        warmups = 0 if workload == "llm_tokens" else 1
        fn = self.run_fn(workload)
        if workload in IMAGES:
            comfy_free()
        run = 0
        for uv in cells:
            self.set_uv(uv)
            for _ in range(warmups):
                self.measure(workload, 0, fn, timed=False)
            for _ in range(timed):
                run += 1
                self.measure(workload, run, fn)

    def start_gemm(self):
        self.gemm = Worker("gemm_worker.py")

    def start_llm(self):
        self.llm = Worker("vllm_worker.py", json.dumps({"model": QWEN_DIR}))

    def stop_workers(self):
        for name in ("gemm", "llm"):
            w = getattr(self, name)
            if w:
                w.close()
                setattr(self, name, None)


def selected(args, names):
    return [n for n in names if not args.only or n in args.only.split(",")]


def cmd_session(args):
    if args.session not in (1, 2, 3):
        raise SystemExit("--session must be 1, 2 or 3")
    expected = "on" if args.session == 2 else "off"
    if args.unlock != expected:
        raise SystemExit(f"session {args.session} is unlock {expected}: sessions run off, on, off")
    cells = args.blocks.split(",") if args.blocks else (["off", "off"] if args.session == 3 else ["off", "on", "on", "off"])
    label = args.label or f"session{args.session}"
    run_matrix(args, label, cells, heat=True)


def run_matrix(args, label, cells, heat):
    b = Bench(args, label)
    check_unlock(args)
    with open(os.path.join(b.out, "env.json"), "w") as f:
        json.dump(collect_env(args), f, indent=2)
    log(f"results -> {b.out}; cells {cells}")
    t0 = time.time()
    try:
        gemm, llm, images = selected(args, GEMM), selected(args, LLM), selected(args, IMAGES)
        if gemm or llm or heat:
            log("stopping ComfyUI: one GPU user at a time")
            comfy_down()
        if heat or gemm:
            b.start_gemm()
        if heat:
            b.set_uv(cells[0])
            b.measure("heat_soak", 0, lambda: {**b.gemm.call(op="heat", seconds=args.heat_seconds), "throughput": 0.0}, timed=False)
        for w in gemm:
            b.blocks(w, cells)
        b.stop_workers()
        if llm:
            b.start_llm()
            for w in llm:
                b.blocks(w, cells)
            b.stop_workers()
        if images:
            comfy_up()
            for w in images:
                b.blocks(w, cells)
    finally:
        b.restore()
    log(f"done in {(time.time() - t0) / 60:.1f} min; failures: {b.failures or 'none'}")
    subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), os.path.join(b.out, "runs.csv")])
    return b


def check_unlock(args):
    loaded = unlock_loaded()
    if (args.unlock == "on") != loaded and not args.ignore_unlock_check:
        raise SystemExit(f"--unlock {args.unlock} but gpu_unlock is {'loaded' if loaded else 'not loaded'} "
                         "(--ignore-unlock-check to override)")


def cmd_stock_ref(args):
    """Each hashed workload twice at unlock off / undervolt off; the reference is
    written only from runs that agree with each other."""
    args.unlock = "off"
    args.timed = args.token_runs = 2
    args.only = args.only or ",".join(HASHED)
    b = run_matrix(args, args.label or "stock-ref", ["off"], heat=False)
    write_reference(b.out, args.reference)


def write_reference(out, reference):
    ref = {}
    # rerunning with --only replaces just those workloads
    if os.path.exists(reference):
        with open(reference) as f:
            ref = {k: v for k, v in json.load(f).items() if k in HASHED}
    with open(os.path.join(out, "runs.csv")) as f:
        rows = [r for r in csv.DictReader(f)]
    for w in HASHED:
        hashes = [r["output_sha256"] for r in rows if r["workload"] == w]
        if not hashes:
            continue
        ref[w] = {"sha256": hashes[0], "deterministic": len(set(hashes)) == 1, "runs": hashes}
    if any(r["workload"] == "llm_tokens" for r in rows):
        with open(os.path.join(out, "llm_tokens.jsonl")) as f:
            ref["llm_tokens"]["tokens"] = json.loads(f.readline())["tokens"]
    for w in HASHED:
        if any(r["workload"] == w for r in rows):
            ref[w]["source"] = os.path.relpath(out, HERE)
            ref[w]["boot_id"] = next(r["boot_id"] for r in rows if r["workload"] == w)
            ref[w]["created"] = datetime.datetime.now().astimezone().isoformat()
    bad = [w for w in ref if not ref[w]["deterministic"]]
    with open(reference, "w") as f:
        json.dump(ref, f, indent=2)
    log(f"reference written to {reference}; non-deterministic at stock: {bad or 'none'}")


def cmd_soak(args):
    if not os.path.exists(args.reference):
        raise SystemExit(f"no reference at {args.reference}; run stock-ref first")
    check_unlock(args)
    b = Bench(args, args.label or f"soak-{datetime.datetime.now():%Y%m%d-%H%M}")
    with open(os.path.join(b.out, "env.json"), "w") as f:
        json.dump(collect_env(args), f, indent=2)
    end = time.time() + args.hours * 3600
    run = 0
    try:
        comfy_up()
        b.set_uv(args.undervolt)
        while time.time() < end:
            for w in IMAGES:
                # the first SD 1.5 batch after a model (re)load hashes differently
                # from every later one, even at stock, so it is an unjudged warm-up
                b.measure(w, 0, b.run_fn(w), timed=False)
                for _ in range(args.chunk):
                    run += 1
                    b.measure(w, run, b.run_fn(w))
    except Abort as e:
        log(f"SOAK ABORTED: {e}")
        raise
    finally:
        b.restore()
        log(f"soak: {run} batches, mismatches: {b.failures or 'none'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("session", "stock-ref", "soak"):
        s = sub.add_parser(name)
        s.add_argument("--label", help="results/<label>/ (default: session<N>, stock-ref, soak-<time>)")
        s.add_argument("--reference", default=os.path.join(HERE, "results", "reference.json"))
        s.add_argument("--force", action="store_true", help="append to an existing results dir")
        s.add_argument("--only", help="comma-separated workloads: " + ",".join(GEMM + LLM + IMAGES))
        s.add_argument("--ignore-unlock-check", action="store_true")
        s.add_argument("--timed", type=int, default=5)
        s.add_argument("--token-runs", type=int, default=1, help="token-set runs per block")
        s.add_argument("--prefill-prompts", type=int, default=200)
        s.add_argument("--heat-seconds", type=int, default=180)
        s.add_argument("--quick", action="store_true",
                       help="dry run: 1 timed per block, cells off,on, 10 prefill prompts, 30 s heat soak")
        if name == "session":
            s.add_argument("--session", type=int, required=True)
            s.add_argument("--unlock", choices=["on", "off"], required=True)
            s.add_argument("--blocks", help="override the cell order, e.g. off,on,on,off")
        if name == "soak":
            s.add_argument("--unlock", choices=["on", "off"], required=True)
            s.add_argument("--undervolt", choices=["on", "off"], default="on")
            s.add_argument("--hours", type=float, default=2.0)
            s.add_argument("--chunk", type=int, default=10, help="judged batches per model before switching")
    args = p.parse_args()
    if getattr(args, "unlock", None) is None:
        args.unlock = "off"
    if args.quick:
        args.timed, args.prefill_prompts, args.heat_seconds = 1, 10, 30
        if args.mode == "session" and not args.blocks:
            args.blocks = "off,on"
    if os.geteuid() != 0:
        raise SystemExit("run as root: the runner drives systemctl and nvidia-smi")
    try:
        {"session": cmd_session, "stock-ref": cmd_stock_ref, "soak": cmd_soak}[args.mode](args)
    except Abort as e:
        log(f"ABORTED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
