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

#dissl_resnet50_dNone_e400_m6,dissl_resnet50_dNone_e100_m2,dissl_resnet50_dNone_e400_m2

kwargs_multi="
representor=dissl_resnet50_d8192_e100_m2,dissl_resnet50_d8192_e400_m6,dissl_resnet50_d8192_e800_m8
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

