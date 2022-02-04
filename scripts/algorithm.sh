#!/usr/bin/env bash


experiment="algorithm"
notes="**Goal**: plot all the radar charts."


# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=simclr_rn50,swav_rn50,clip_rn50,dino_rn50,barlow_rn50,mocov2_rn50
"

kwargs_multi="
representor=clip_rn50
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