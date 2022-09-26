#!/usr/bin/env bash

experiment="sigterm"
notes="**Goal**: debug script."

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
predictor=torch_linear_hypopt_test
predictor.hypopt.n_hyper=3
predictor.hypopt.kwargs_sampler.n_startup_trials=2
trainer.max_epochs=3
"
#torch_linear_hypopt

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python sigterm.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m

  done
fi

