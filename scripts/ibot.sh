#!/usr/bin/env bash

experiment="vit"
notes="**Goal**: evaluate all the ibot models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=ibot_vitB16,ibot_vitS16,ibot_vitL16
seed=123
predictor=torch_linear
"

kwargs_multi="
representor=ibot_vitL16
seed=123
predictor=torch_linear
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

