#!/usr/bin/env bash

experiment="vissl"
notes="**Goal**: evaluate all the vissl models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# can run npidpp_rn50w2
# once https://github.com/facebookresearch/vissl/issues/516

kwargs_multi="
representor=simclr_rn50w4
seed=123
predictor=torch_linear_delta_hypopt
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

