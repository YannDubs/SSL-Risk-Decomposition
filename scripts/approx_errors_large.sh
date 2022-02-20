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



# run on large server
kwargs_multi="
representor=sup_rn50w2,sup_vitL16,sup_vitL16_beit
"

# run on large server
kwargs_multi="
representor=sup_vitB8_dino
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