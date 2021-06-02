#!/bin/bash
OUTDIR=outputs/permuted_MNIST_incremental_domain
REPEAT=5
source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

mkdir -p $OUTDIR
python -u ../iBatchLearn.py --repeat $REPEAT --optimizer SGD  --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 2 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name NASH_16k  --lr 0.1 --reg_coef 0.5  | tee ${OUTDIR}/Nash.log
