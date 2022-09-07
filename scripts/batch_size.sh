#!/usr/bin/env bash

experiment="batch_size"
notes="**Goal**: compare simclr trained with different batch sizes."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=simclr_rn50_bs512_ep100,simclr_rn50_bs4096_ep100,swav_rn50_ep400_bs256,swav_rn50_ep200_bs256
seed=123,124,125
"
# add swav_rn50_ep400,swav_rn50_ep200 from other runs



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior