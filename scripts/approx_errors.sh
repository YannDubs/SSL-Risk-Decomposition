#!/usr/bin/env bash

experiment="approx_errors"
notes="**Goal**: compute approx errors for common architectures."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# need to add imagenet22k
kwargs_multi="
representor=sup_rn50,sup_rn101,sup_vitB16,sup_vitS16,sup_vitB32
seed=123,124,125
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior