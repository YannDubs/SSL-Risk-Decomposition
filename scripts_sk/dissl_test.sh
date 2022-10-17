#!/usr/bin/env bash

experiment="dissl"
notes="**Goal**: evaluate all the models we pretrained."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=dissl_resnet50_dNone_e100_m2
seed=123
predictor=sk_logistic_hypopt_test
data.kwargs.is_avoid_raw_dataset=True
data.subset=0.01
data.kwargs.subset_raw_dataset=0.3
"

# need to run seed=124,125
# torch_linear_erm

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m

  done
fi

