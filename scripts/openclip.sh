#!/usr/bin/env bash

experiment="openclip"
notes="**Goal**: evaluate all the openclip models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=openclip_vitB32,openclip_vitL14,openclip_vitH14,openclip_vitG14,openclip_vitG14_extractPred,openclip_vitG14_extractS,openclip_vitG14_extractB
seed=123
predictor=torch_linear_delta_hypopt
"

kwargs_multi="
representor=openclip_vitH14_extractPred,openclip_vitH14_extractS,openclip_vitH14_extractB
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

