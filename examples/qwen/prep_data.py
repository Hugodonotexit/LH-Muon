"""
prep_data.py -- general-text data for the continued-pretraining optimizer tests (Qwen3.5-0.8B-Base,
GPT-2), tokenized with the chosen model's tokenizer.

Reads raw Dolma v1.7 json.gz files (one {"text": ...} per line) and writes, per source, a flat uint32
token file (uint32: Qwen's vocabulary is 248K, past uint16) of documents separated by EOS, plus eval.npy:
[64, 1025] windows cut from documents that never enter the training files.

    python examples/qwen/prep_data.py --out data/qwen_mix
    python examples/qwen/prep_data.py --model openai-community/gpt2 --tokens 12000000 --out data/gpt2_mix
"""

import argparse
import gzip
import json
import os

import numpy as np
from transformers import AutoTokenizer

SOURCES = {           # source: (files, training tokens for the default --tokens scale of 1.0)
    "c4": (["c4-0000.json.gz"], 5_000_000),
    "cc": (["cc_en_head-0000.json.gz"], 5_000_000),
    "books": (["books-0002.json.gz"], 4_000_000),
    "wiki": ([f"megawika-{i:04d}.json.gz" for i in range(60)], 3_000_000),
    "arxiv": (["arxiv-0000.json.gz"], 3_000_000),
}
EVAL_PER_SOURCE = 13          # 5 x 13 = 65 >= 64 windows


def docs(paths):
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                t = json.loads(line).get("text", "")
                if len(t) > 200:
                    yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/hugo/dolma_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--tokens", type=int, default=None, help="training tokens per source (default: the per-source table)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.eos_token_id
    evals, manifest = [], {}
    for name, (files, target) in SOURCES.items():
        target = a.tokens or target
        it = docs([os.path.join(a.raw, f) for f in files])
        # eval first: whole documents >= 1025 tokens, never reused for training
        got = 0
        while got < EVAL_PER_SOURCE:
            ids = tok(next(it))["input_ids"]
            if len(ids) >= 1025:
                evals.append(np.asarray(ids[:1025], dtype=np.uint32))
                got += 1
        buf, n = [], 0
        while n < target:
            batch = [next(it) for _ in range(16)]            # small batches: whole books overshoot less
            for ids in tok(batch)["input_ids"]:
                ids.append(eos)
                buf.append(np.asarray(ids, dtype=np.uint32))
                n += len(ids)
        arr = np.concatenate(buf)
        path = os.path.join(a.out, f"{name}.bin")
        arr.tofile(path)
        manifest[name] = {"path": path, "tokens": int(arr.size)}
        print(f"{name}: {arr.size / 1e6:.1f}M training tokens, {EVAL_PER_SOURCE} eval windows", flush=True)
    ev = np.stack(evals)
    # interleave by source so any prefix covers every source
    order = [s * EVAL_PER_SOURCE + i for i in range(EVAL_PER_SOURCE) for s in range(len(SOURCES))]
    np.save(os.path.join(a.out, "eval.npy"), ev[order][:64])
    json.dump({"model": a.model, "sources": manifest, "eval_sources": list(SOURCES)}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
