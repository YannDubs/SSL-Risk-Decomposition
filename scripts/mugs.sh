#!/usr/bin/env bash

experiment="mugs"
notes="**Goal**: evaluate all the mugs models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=mugs_vits16_ep100,mugs_vits16_ep300,mugs_vits16_ep800,mugs_vitb16_ep400,mugs_vitl16_ep250,mugs_vits16_ep800_extractS,mugs_vitb16_ep400_extractB
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

