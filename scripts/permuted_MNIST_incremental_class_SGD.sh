#!/bin/bash

OUTDIR=../outputs/permuted_MNIST_incremental_class
REPEAT=5
mkdir -p $OUTDIR


source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"


python -u ../iBatchLearn.py --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule 2 --batch_size 128 --model_name MLP1000                                                     --lr 0.001                       | tee ${OUTDIR}/SGD.log
