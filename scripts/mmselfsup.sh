#!/usr/bin/env bash

experiment="mmselfsup"
notes="**Goal**: evaluate all the mmselfsup models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=relativeloc_rn50_70ep_mmselfsup,odc_rn50_440ep_mmselfsup,densecl_rn50_200ep_mmselfsup,simsiam_rn50_bs256_ep200_mmselfsup,simclr_rn50_bs256_ep200_mmselfsup,deepcluster_rn50_bs512_ep200_mmselfsup
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

