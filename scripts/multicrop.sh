#!/usr/bin/env bash

experiment="multicrop"
notes="**Goal**: compare effect of multicrop."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=dc2_rn50_ep400_2x224+4x96,dc2_rn50_ep400_2x224,swav_rn50_ep400_2x224
seed=123
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait
