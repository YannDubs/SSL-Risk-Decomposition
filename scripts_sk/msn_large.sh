#!/usr/bin/env bash

experiment="msn"
notes="**Goal**: evaluate all the MSN models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=msn_vitb4_ep300_extracts,msn_vitb4_ep300_extractb,msn_vitl7_ep200_extractb,msn_vitl7_ep200_extracts
seed=123
predictor=sk_logistic_hypopt
data.kwargs.is_avoid_raw_dataset=True
data.kwargs.subset_raw_dataset=0.3
"

kwargs_multi="
representor=msn_vitb4_ep300_extracts,msn_vitb4_ep300_extractb
seed=123
predictor=sk_logistic_hypopt
data.kwargs.is_avoid_raw_dataset=True
data.kwargs.subset_raw_dataset=0.3
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

