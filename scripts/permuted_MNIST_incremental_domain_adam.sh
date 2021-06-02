#!/bin/bash

GPUID=1
OUTDIR=outputs/permuted_MNIST_incremental_domain
REPEAT=5
source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

mkdir -p $OUTDIR
python -u ../iBatchLearn.py  --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 2 --batch_size 128 --model_name MLP1000                                                     --lr 0.0001                      | tee ${OUTDIR}/Adam.log