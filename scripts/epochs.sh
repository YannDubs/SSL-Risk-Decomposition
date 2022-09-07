#!/usr/bin/env bash

experiment="epochs"
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
representor=swav_rn50_ep100,swav_rn50_ep200,swav_rn50_ep400,simclr_rn50_ep200,simclr_rn50_ep400,simclr_rn50_ep800,simclr_rn50w2_ep100,simclr_rn101_ep100,pirl_rn50_ep200,barlow_rn50_ep300,mocov3_rn50_ep300,mocov3_rn50_ep1000
seed=123,124,125
"
# also add swav_rn50,simclr_rn50_bs4096_ep100,simclr_rn50,simclr_rn50w2,simclr_rn101,pirl_rn50 from other


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior