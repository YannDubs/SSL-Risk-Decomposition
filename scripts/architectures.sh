#!/usr/bin/env bash

experiment="architectures"
notes="**Goal**: compare swav trained for a different number of epochs."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over


kwargs_multi="
seed=124,125
predictor=torch_linear
representor=simclr_rn50w4
" # RUNNING (to run seeds)

kwargs_multi="
seed=123
predictor=torch_linear
representor=swav_rn50w2,swav_rn50w4,swav_rn50w5
data.kwargs.num_workers=0
hydra.launcher.mem_gb=50
" # RUNNING (to run seeds)
# waitign for npidpp_rn50w2
# careful with memory isses
# /juice/scr/yanndubs/SSL-Risk-Decomposition/outputs/2022-02-21_13-19-02

kwargs_multi="
seed=123,124,125
predictor=torch_linear
representor=simclr_rn101,dino_vitB8_last,dino_vitB16_last,dino_vitS16_last,clip_rn101,clip_vitB32,clip_vitB16,clip_rn50x4,clip_rn50x16,clip_rn50x64,simclr_rn50w4,swav_rn50w2,swav_rn50w4,swav_rn50w5,npidpp_rn50w2
"
# also add simclr_rn50w2, simclr_rn50,swav_rn50_ep400 from other experiments
# clip_rn50x16,clip_rn50x64 should be ran on large GPUs





if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior