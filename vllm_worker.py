"""vLLM worker: loads Qwen3-8B once and answers JSON commands on stdin with
'@@RESULT {json}' lines on stdout. The timed path is vllm bench throughput's own
(RandomDataset + _run_vllm_requests), so numbers match `vllm bench throughput`
without reloading the model for every run."""
import hashlib
import json
import sys
import time

import vllm
from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import RandomDataset
from vllm.benchmarks.throughput import _run_vllm_requests

TOKEN_PROMPTS = [
    "The capital of France is",
    "Explain in one paragraph why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List the planets of the solar system in order from the sun:",
    "Translate to Spanish: The quick brown fox jumps over the lazy dog.",
    "A train leaves at 3 pm travelling 80 km/h. How far has it gone by 5:30 pm?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What is the derivative of x^3 * sin(x)?",
    "Give three tips for writing clear technical documentation.",
    "def quicksort(arr):",
    "The main differences between TCP and UDP are",
    "Once upon a time, in a village at the edge of a forest,",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "Describe how a transformer neural network processes a sentence.",
    "Buenos Aires is a city known for",
    "The first ten prime numbers are",
    "Write a haiku about a graphics card running hot.",
    "In Rust, ownership means that",
    "Convert 72 degrees Fahrenheit to Celsius and show the steps.",
    "The three laws of thermodynamics state that",
]


def main():
    cfg = json.loads(sys.argv[1])
    llm = LLM(
        model=cfg["model"],
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        seed=0,
        # one process serves every run; with the cache on, the second identical
        # prefill run would reuse the first one's KV instead of computing it
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    print("@@RESULT " + json.dumps({"ready": True, "vllm": vllm.__version__}), flush=True)

    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd["op"] == "random":
            # fresh dataset each call: every run sees the same prompts
            reqs = RandomDataset(random_seed=0).sample(
                tokenizer=tokenizer,
                num_requests=cmd["num_prompts"],
                input_len=cmd["input_len"],
                output_len=cmd["output_len"],
            )
            seconds, outputs = _run_vllm_requests(
                llm, reqs, n=1, disable_detokenize=False, do_profile=False,
                prequeue_requests=False, enable_lora=False,
            )
            prompt_toks = sum(len(o.prompt_token_ids) for o in outputs)
            out_toks = sum(len(c.token_ids) for o in outputs for c in o.outputs)
            res = {"seconds": seconds, "prompt_tokens": prompt_toks, "output_tokens": out_toks}
        elif cmd["op"] == "tokens":
            sp = SamplingParams(temperature=0.0, max_tokens=cmd["max_tokens"], seed=0)
            tokens = []
            t0 = time.perf_counter()
            for p in TOKEN_PROMPTS:
                out = llm.generate([p], sp, use_tqdm=False)
                tokens.append(list(out[0].outputs[0].token_ids))
            seconds = time.perf_counter() - t0
            res = {
                "seconds": seconds,
                "output_tokens": sum(map(len, tokens)),
                "tokens": tokens,
                "sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            }
        else:
            res = {"error": f"unknown op {cmd['op']}"}
        print("@@RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
