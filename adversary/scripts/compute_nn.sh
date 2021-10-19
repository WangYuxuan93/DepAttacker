#!/usr/bin/env bash
main=./scripts/build_embed_nn.py
input=[adjusted embedding]
output=[output path]

#source /users2/yxwang/work/env/py3.6/bin/activate
int=$1
thread=$2
if [ -z $2 ];then
  echo '[int] [n_thread]'
  exit
fi

python $main $input $output --nn --k 101 --log_every 1000 --n_thread $thread --interval $int