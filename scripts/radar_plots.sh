#!/usr/bin/env bash

experiment="radar_plots"
notes="**Goal**: plot all the radar charts."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# run on large server
kwargs_multi="
seed=123,124,125
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "representor=clip_vitL14,simclr_rn50w2,beit_vitL16 predictor=torch_linear" "dino_vitB8 predictor=torch_linear_dino"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior