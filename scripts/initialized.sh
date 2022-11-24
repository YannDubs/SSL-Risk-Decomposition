#!/usr/bin/env bash

experiment="initialized"
notes="**Goal**: evaluate representations initialization of the model (ie no SSL)."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=init_rn50,init_rn50_d4096,init_rn50_d8192,init_rn50_d1024,init_rn50_d512,init_rn101,init_rn50w2,init_vitB8,init_vitB16,init_vitB16_dino,init_vitB16_dino_extractS,init_vitB32,init_vitL16,init_vitS16,init_vitS16_dino,init_vitS16_dino_extractB
seed=123
predictor=torch_linear_delta_hypopt
"

kwargs_multi="
representor=init_vitB8_dino,init_rn50_d8192
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

