#!/usr/bin/env bash


experiment="inv_relabelings_k2"
notes="**Goal**: understand whether you can predict any invariant task."


# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
predictor=torch_logistic
rand_task.n_rand_tasks=10000
"

kwargs_multi="
representor=simclr_rn50,clip_rn50x64,swav_rn50w2
seed=123
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python main_rand_task.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior