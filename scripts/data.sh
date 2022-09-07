#!/usr/bin/env bash

experiment="data"
notes="**Goal**: understand effect of training data."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# should run also colorization and jigsaw as a comparinson

# every arguments that you are sweeping over
kwargs_multi="
representor=rotnet_rn50_in1k,rotnet_rn50_in22k
seed=123,124,125
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior