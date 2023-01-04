#!/usr/bin/env bash

experiment="statistics"
notes="**Goal**: estimate all the desired statistics of the representations."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


kwargs_multi="
representor=barlow_rn50
seed=123
data=imagenet
data.kwargs.subset_raw_dataset=0.05
data.kwargs.is_avoid_raw_dataset=True
statistics.force_recompute=True
statistics.n_augs=3
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python statistics.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs

    sleep 10

  done
fi

