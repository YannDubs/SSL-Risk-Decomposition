#!/usr/bin/env bash

experiment="approx_errors"
notes="**Goal**: compute approx errors for differnt way's of extracting features for vit."

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


kwargs_multi="
representor=sup_vitS16_dino,sup_vitB16_dino,sup_vitB16_dino_extractS,sup_vitS16_dino_extractB,sup_vitB16,sup_vitS16
predictor=torch_linear_dino,torch_linear
seed=123,124,125
"




if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior