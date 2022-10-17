#!/usr/bin/env bash

experiment="mae"
notes="**Goal**: evaluate all the MAE models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=mae_vitB16,mae_vitL16,mae_vitH14
seed=123
predictor=torch_linear_hypopt
"

kwargs_multi="
representor=mae_vitB16,mae_vitL16,mae_vitH14
seed=123
predictor=torch_linear_lr
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

