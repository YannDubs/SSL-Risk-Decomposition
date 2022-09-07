#!/usr/bin/env bash

experiment="predictor_vit"
notes="**Goal**:understand effect of predictor on vit."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# run on large server
kwargs_multi="
representor=dino_vitS16,dino_vitB16,dino_vitB16_extractS,dino_vitS16_extractB
predictor=torch_linear_dino,torch_linear
seed=123,124,125
"
# add dino_vitB16_last,dino_vitS16_last
# run seed 124,125 once happy



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior