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





# simclr_rn50w4 needs XL
kwargs_multi="
representor=simclr_rn50w4
seed=123
predictor=sk_logistic_hypopt
data.kwargs.is_avoid_raw_dataset=True
data.subset=0.01
data.kwargs.subset_raw_dataset=0.3
"


# torch_linear_hypopt

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

