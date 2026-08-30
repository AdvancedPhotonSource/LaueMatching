#!/bin/bash
# launch_shard.sh SHARD GPU PORT NCPUS
# One LaueMatching orchestrator for one shard, on whatever host it is invoked on.
# Runs detached; writes a per-shard log under logs/.
set -u
K=$1; GPU=$2; PORT=$3; NC=${4:-16}
W=$LAUE_WORK
LM=/home/beams/EPIX34ID/opt/LaueMatching
PY=/home/beams/EPIX34ID/conda-envs/laue_rt/bin/python
H=$(hostname -s)

mkdir -p $W/logs $W/results
cd $W

WORK=$W PY=$PY WATCH="" NCPUS=$NC \
  ALPHA_CONFIG=$W/params/params_Zn_h_s$K.txt BETA_CONFIG="" \
  ALPHA_GPU=$GPU ALPHA_PORT=$PORT \
  bash $LM/pipeline/run_laue.sh $W/shards/h_s$K /entry1/data/data \
  > $W/logs/shard${K}_${H}.launch 2>&1

sleep 8
echo "host=$H shard=$K gpu=$GPU port=$PORT ncpus=$NC"
pgrep -af "laue_orchestrator.py --config $W/params/params_Zn_h_s$K.txt" | head -1
tail -2 $W/logs/shard${K}_${H}.launch
