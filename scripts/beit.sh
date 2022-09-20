#!/usr/bin/env bash

experiment="beit"
notes="**Goal**: evaluate all the BEIT models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=beit_vitB16_pt22k,beit_vitL16_pt22k,beitv2_vitB16_pt1k,beitv2_vitL16_pt1k,beitv2_vitB16_pt1k_extractB
seed=123
predictor=torch_linear
"

kwargs_multi="
representor=beitv2_vitB16_pt1k_ep300
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

