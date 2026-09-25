#!/usr/bin/env bash
# GPT-2 optimizer test, one run at a time on the RTX 3060. Finished runs are skipped.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=2 PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p runs/gpt2
for run in "lhmuon 3e-4" "muon 3e-4" "adamw 3e-4" "lion 3e-5" \
           "lhmuon 1e-4" "muon 1e-4" "adamw 1e-4" "lion 1e-5" \
           "lhmuon 3e-5" "muon 3e-5" "adamw 3e-5"; do
  set -- $run
  [ -f runs/gpt2/${1}_lr$2/final.json ] && continue
  echo "$(date +%H:%M:%S) $1 lr $2"
  python examples/gpt2/optimizer_test.py --opt $1 --lr $2 --out runs/gpt2/${1}_lr$2 > runs/gpt2/${1}_lr$2.stdout 2>&1 || echo "FAILED $1 $2"
  grep '^step 600' runs/gpt2/${1}_lr$2.stdout
done
python examples/qwen/analyze.py runs/gpt2 > /dev/null && echo "queue finished; report in runs/gpt2/report.md"
