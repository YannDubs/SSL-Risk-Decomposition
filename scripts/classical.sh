#!/usr/bin/env bash

experiment="classical"
notes="**Goal**: compare classical (pre simclr) methods."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=jigsaw_rn50,npid_rn50,clusterfit_rn50,pirl_rn50,npidpp_rn50
seed=123,124,125
"
# should add colorization


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior