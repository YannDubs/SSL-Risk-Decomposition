#!/usr/bin/env bash

# parses special mode for running the script
source `dirname $0`/utils.sh

experiment=$prfx"radar_plots"
notes="**Goal**: plot all the radar charts."

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=7200
"

# run on large server
kwargs_multi="
representor=beit_vitL16,dino_vitB8,clip_vitL14,simclr_rn50w2
"
# add swav_rn50 from other experiments

kwargs_multi="
representor=clip_vitL14,simclr_rn50w2
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# ADD plotting behavior