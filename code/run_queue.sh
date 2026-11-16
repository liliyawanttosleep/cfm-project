#!/bin/bash
# Resumable GPU queue for the final-report experiments. Each step skips itself when its output exists;
# the Python scripts also resume inside a step.
cd /c/Users/Wentao/Desktop/cfm_project
PY=D:/Anaconda/envs/flow/python.exe
run() { local log=$1; shift; echo "=== $(date +%m-%d\ %T) start: $*"; "$PY" -u "$@" >> "$log" 2>&1; echo "=== $(date +%m-%d\ %T) exit $?"; }
train() { # method seed extra-args... ; skip if final ckpt exists
  local m=$1 tag=$2 log=$3; shift 3
  if [ -f results/e2/ckpt/${m}_s0${tag}.pt ]; then echo "skip train ${m}${tag}"; else run "$log" code/e2_mnist/train_mnist.py --method $m --seed 0 "$@"; fi; }
evalm() { local m=$1 tag=$2 log=$3; run "$log" code/e3_kmeans/fid_seed.py --seed 0 --tag "$tag" --methods $m --dopri; }

# C5 reflow from I-CFM (required)
run results/c5_reflow/logs/pairs_icfm_s0.log code/c5_reflow/make_pairs.py --method icfm --seed 0
train icfm _reflow results/c5_reflow/logs/train_icfm_s0_reflow.log --reflow
evalm icfm _reflow results/c5_reflow/logs/fid_icfm_s0_reflow.log
# C2 check on seeds 1 and 2
L=results/c2_solvers/logs
for s in 1 2; do for m in otcfm icfm; do
  run $L/${m}_s$s.log code/c2_solvers/sample_grid.py --method $m --seed $s --core
  run $L/t1_${m}_s$s.log code/c2_solvers/t1_check.py --method $m --seed $s
done; done
# C3 OT batch
for B in 512 2048; do
  train otcfm _otb$B results/c3_otbatch/logs/train_otcfm_s0_otb$B.log --ot-batch $B
  evalm otcfm _otb$B results/c3_otbatch/logs/fid_otcfm_s0_otb$B.log
done
# C4 t distribution
for D in logitnormal kappa; do
  train icfm _t$D results/c4_tdist/logs/train_icfm_s0_t$D.log --t-dist $D
  evalm icfm _t$D results/c4_tdist/logs/fid_icfm_s0_t$D.log
done
# C5 step 4: reflow from OT-CFM
run results/c5_reflow/logs/pairs_otcfm_s0.log code/c5_reflow/make_pairs.py --method otcfm --seed 0
train otcfm _reflow results/c5_reflow/logs/train_otcfm_s0_reflow.log --reflow
evalm otcfm _reflow results/c5_reflow/logs/fid_otcfm_s0_reflow.log
# C6: do inference-side gains add to reflow? (solver x grid on the reflow models)
for m in icfm otcfm; do
  [ -f results/e2/ckpt/${m}_s0_reflow.pt ] && run results/c2_solvers/logs/${m}_s0_reflow.log code/c2_solvers/sample_grid.py --method $m --seed 0 --tag _reflow --core
done
echo "ALL DONE $(date +%m-%d\ %T)"
