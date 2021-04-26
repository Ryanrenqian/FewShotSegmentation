#!/bin/sh
PARTITION=Segmentation
dataset=$1
model_name=$2
exp_name=$3
exp_din=exp/${model_name}/${dataset}/${exp_name}
model_din=${exp_din}/model
result_din=${exp_din}/result
config=data/config/${dataset}/${dataset}_${model_name}_${exp_name}.yaml
mkdin -p ${model_din} ${result_din}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/test.sh tool/test.py ${config} ${exp_din}
python -u -m tool.test --config=${config} 2>&1 | tee ${result_din}/test-$now.log

