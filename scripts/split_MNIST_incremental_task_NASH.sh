#!/bin/bash

OUTDIR=outputs/split_MNIST_incremental_task
REPEAT=1

source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

mkdir -p $OUTDIR
python -u ../iBatchLearn.py  --repeat $REPEAT --optimizer SGD     --force_out_dim 0 --first_split_size 5 --other_split_size 5 --schedule 2 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name NASH_16k --lr 0.1 --reg_coef 0.0001  | tee ${OUTDIR}/NASH_16k.log
