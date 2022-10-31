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
representor=dissl_resnet50_dNone_e100_m2,dissl_resnet50_dNone_e400_m2,dissl_resnet50_dNone_e400_m6
seed=123
predictor=torch_linear_delta_hypopt
"

kwargs_multi="
representor=dissl_resnet50_dNone_e100_m2
seed=123
predictor=torch_linear_delta_hypopt
"


# need to run seed=124,125
# torch_linear_erm

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

