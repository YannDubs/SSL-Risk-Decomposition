#!/usr/bin/env bash

experiment="moco"
notes="**Goal**: evaluate all the MOCO models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=mocov3_rn50_ep100,mocov3_rn50_ep300,mocov3_rn50_ep1000,mocov3_vitS_ep300,mocov3_vitB_ep300,mocov1_rn50_ep200,mocov2_rn50_ep200,mocov2_rn50_ep800
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

