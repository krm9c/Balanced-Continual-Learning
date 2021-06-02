#!/bin/bash


GPUID=$1
OUTDIR=outputs/permuted_MNIST_incremental_task
REPEAT=5
source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

mkdir -p $OUTDIR
python -u ../iBatchLearn.py --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 0.01        | tee ${OUTDIR}/MAS.log
