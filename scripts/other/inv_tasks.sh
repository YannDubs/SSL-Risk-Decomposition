#!/usr/bin/env bash


experiment="inv_tasks"
notes="**Goal**: understand whether you can predict any invariant task."


# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
predictor=torch_logistic
rand_task.n_rand_tasks=10
rand_task.n_label_per_task=1000
data.kwargs.is_add_idx=True
rand_task.components=[train-sbst-0.1_train-sbst-0.1]
"


kwargs_multi="
representor=simclr_rn50
seed=0
rand_task.n_runs=10
" # 3681223

kwargs_multi="
representor=swav_rn50w2
seed=0
rand_task.n_runs=10
" # inv_tasks

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python main_rand_task.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# ADD plotting behavior