#!/usr/bin/env bash

experiment="approx_errors"
notes="**Goal**: compute approx errors for different predictor architectures."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


kwargs_multi="
representor=sup_rn50
predictor=torch_mlp
seed=123,124,125
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" "predictor.arch_kwargs.n_hid_layers=1,3" "predictor.arch_kwargs.hid_dim=256,4096"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior