#!/usr/bin/env bash
main=./scripts/merge_embed_nn.py
input=iput path
output=output path
# vocab的长度
size=185050

#source /users2/yxwang/work/env/py3.6/bin/activate
python $main $input $output --size $size