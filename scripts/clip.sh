#!/usr/bin/env bash

experiment="clip"
notes="**Goal**: evaluate all the clip models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


kwargs_multi="
representor=clip_rn50,clip_rn50x4,clip_rn50x16,clip_rn50x64,clip_rn101,clip_vitB16,clip_vitB32,clip_vitL14,clip_vitL14_px336
seed=123
predictor=torch_linear_hypopt
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

