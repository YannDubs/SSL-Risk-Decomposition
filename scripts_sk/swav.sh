#!/usr/bin/env bash

experiment="swav"
notes="**Goal**: evaluate all the swav models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=swav_rn50,swav_rn50_ep100,swav_rn50_ep200,swav_rn50_ep200_bs256,swav_rn50_ep400,swav_rn50_ep400_2x224,swav_rn50_ep400_bs256,dc2_rn50_ep400_2x224,dc2_rn50_ep400_2x160_4x96,dc2_rn50_ep800_2x224_6x96,selav2_rn50_ep400_2x224,selav2_rn50_ep400_2x160_4x96
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

