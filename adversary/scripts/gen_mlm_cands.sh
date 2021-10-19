#!/usr/bin/env bash
lmdir=/users2/yxwang/work/data/models
#lmpath=$lmdir/bert-base-uncased
#lmpath=$lmdir/bert-large-cased
lmpath=$lmdir/bert-large-uncased

export PYTHONPATH=../../

if [ -z $3 ];then
  echo '[gpu] [input] [output]'
  exit
fi

#dir=/users2/yxwang/work/data/ptb/dependency-stanford-chen14
#train=$dir/PTB_train_auto.conll
#dev=$dir/PTB_dev_auto.conll
#test=$dir/PTB_test_auto.conll
#demo=data/dev.conll
num=50

gpu=$1
input=$2
output=$3

main=/users7/zllei/blackbox-attack/adversary/scripts/gen_mlm_cands.py

# source /users2/yxwang/work/env/py3.6/bin/activate
CUDA_VISIBLE_DEVICES=$gpu python $main --input $input --output $output --bert_path $lmpath --temp 1.0 --top_k 100 --n_mlm_cands $num
