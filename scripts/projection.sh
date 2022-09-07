#!/usr/bin/env bash

experiment="projection"
notes="**Goal**: compare effect of projection head."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=pirl_rn50,pirl_rn50_ep200,pirl_rn50_headMLP,pirl_rn50_ep200_headMLP
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
