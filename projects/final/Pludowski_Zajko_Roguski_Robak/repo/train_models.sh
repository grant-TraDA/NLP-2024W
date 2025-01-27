#!/bin/bash

set -e

seeds=(1)
models=(roberta ernie)
datasets=(liar coaid isot)
masks=(yes no)

counter=0
for seed in ${seeds[@]}; do
    for mask in ${masks[@]}; do
        for model in ${models[@]}; do
            for dataset in ${datasets[@]}; do
                echo "Starting $((++counter)) training - seed: $seed    mask: $mask     model: $model    dataset: $dataset"
                python train.py --model $model --data $dataset --mask $mask --seed $seed
            done
        done
    done
done
