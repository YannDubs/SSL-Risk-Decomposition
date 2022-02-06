#!/usr/bin/env bash

experiment="architectures"
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
representor=jigsaw_rn50,colorization_rn50,npid_rn50,clusterfit_rn50
"
# also add simclr_rn50, rotnet_rn50_in1k

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# ADD plotting behavior