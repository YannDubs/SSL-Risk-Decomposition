#!/usr/bin/env bash

experiment="dino"
notes="**Goal**: evaluate all the dino models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=dino_rn50,dino_vitS16_last,dino_vitS8_last,dino_vitB16_last,dino_vitB8_last,dino_vitS16,dino_vitB16,dino_vitB8,dino_vitS16_extractB,dino_vitB16_extractS
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

