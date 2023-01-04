#!/usr/bin/env bash

experiment="vicregl"
notes="**Goal**: evaluate all the vicregl models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


kwargs_multi="
representor=vicregl_rn50_alpha09,vicregl_rn50_alpha075,vicregl_convnexts_alpha09,vicregl_convnexts_alpha075
seed=123
predictor=torch_linear_delta_hypopt
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

