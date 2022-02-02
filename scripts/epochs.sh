#!/usr/bin/env bash

# parses special mode for running the script
source `dirname $0`/utils.sh

experiment=$prfx"epochs"
notes="**Goal**: compare swav trained for a different number of epochs."

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=swav_rn50_ep100,swav_rn50_ep200,swav_rn50_ep400
"
# also add swav_rn50 from other


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# ADD plotting behavior