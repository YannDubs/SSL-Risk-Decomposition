#!/usr/bin/env bash

experiment="supervised"
notes="**Goal**: evaluate all the supervised models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=sup_rn50
seed=123
predictor=torch_linear_delta_hypopt
"


#predictor=torch_bnlinear
#seed=123,124,125
#"predictor=torch_linear,torch_momlinear" #"predictor.opt_kwargs.lr=3e-3,3e-2" "predictor.opt_kwargs.weight_decay=0,1e-4" "trainer.max_epochs=33,300" "data.kwargs.batch_size=64,1024"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargspirl_rn50_ep200

    sleep 10

  done
fi

