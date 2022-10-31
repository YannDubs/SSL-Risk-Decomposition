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
representor=msn_vits16_ep800,msn_vitb16_ep600,msn_vitl16_ep600,msn_vitl7_ep200,msn_vitb4_ep300
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

