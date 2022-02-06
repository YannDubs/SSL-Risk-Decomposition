#!/usr/bin/env bash

experiment="approx_errors"
notes="**Goal**: compute approx errors for common architectures."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=2880
"

kwargs_multi="
representor=sup_vitS16_dino,sup_vitB16_dino
"

kwargs_multi="
representor=sup_rn50,sup_rn101,sup_vitB16
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