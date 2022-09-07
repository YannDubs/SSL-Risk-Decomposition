#!/usr/bin/env bash


experiment="sample_eff2"
notes="**Goal**: understand how sample efficient are the models."


# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
predictor=torch_logistic
"



kwargs_multi="
representor=simclr_rn50,swav_rn50,clip_rn50,sup_rn50
seed=123
"

kwargs_multi="
representor=clip_vitL14,clip_rn50x64,swav_rn50w2
seed=123
" # run large

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python main_sample_efficiency.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi


# ADD plotting behavior