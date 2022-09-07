#!/usr/bin/env bash


experiment="algorithm"
notes="**Goal**: comparing different SSL algorithms."


# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=simclr_rn50,swav_rn50,clip_rn50,dino_rn50,barlow_rn50,mocov3_rn50_ep1000
predictor=torch_momlinear
seed=123
" # RUNNING

kwargs_multi="
representor=simclr_rn50,swav_rn50,clip_rn50,dino_rn50,barlow_rn50,mocov3_rn50_ep1000
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