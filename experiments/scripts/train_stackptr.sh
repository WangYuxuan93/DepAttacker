#!/usr/bin/env bash
emb=/users2/yxwang/work/data/embeddings/glove/glove.6B.100d.txt.gz
lmdir=/users2/yxwang/work/data/models
dir=/users2/yxwang/work/data/ptb/dependency-stanford-chen14
model_types=glove-666

train=$dir/PTB_train_auto.conll
dev=$dir/PTB_dev_auto.conll
test=$dir/PTB_test_auto.conll

add_path=none
add_flag=adversary
total_num=0  # 0表示全部
ratio=0.1

lans="en"

tcdir=/users2/yxwang/work/experiments/robust_parser/lm/saves
main=/users7/zllei/NeuroNLP2-yx/experiments/parser.py

seed=666
batch=128
evalbatch=128
epoch=1000
patient=100
lr='0.002'
lm=none
#lm=roberta-base
lmpath=$lmdir/electra-base-discriminator
#lmpath=$lmdir/roberta-large
#lm=electra
#lmpath=$lmdir/electra-large-discriminator
#lmpath=$lmdir/electra-base-discriminator

use_elmo=''
#use_elmo=' --use_elmo '
elmo_path=$lmdir/elmo


random_word=''
#random_word=' --use_random_static '
pretrain_word=' --use_pretrained_static '
#pretrain_word=' --use_pretrained_static '
freeze=''
#freeze=' --freeze'
trim=''
#trim=' --do_trim'
#vocab_size=400000
vocab_size=400000

lmlr='2e-5'
#lmlr=0
opt=adamw
#sched=exponential
#decay='0.99999'
sched=step
decay='0.75'
dstep=5000
warmup=500
reset=20
beta1='0.9'
#beta2='0.999'
beta2='0.9'
eps='1e-8'
beam=1
clip='5.0'
l2decay='0'
unk=0
#unk='1.0'
ndigit=''
#ndigit=' --normalize_digits'
losstype=token
evalevery=1
posidx=3
mix=' --mix_datasets'
form=conllx

gpu=$1
mode=train
save=$2
log_file=${save}/log_${mode}_$(date "+%Y%m%d-%H%M%S").txt
if [ -z $1 ];then
  echo '[gpu] [save] [log]'
  exit
fi

if [ ! -f "$log_file" ]; then
  touch "$log_file"
  chmod 777 "$log_file"
fi

#source /users2/yxwang/work/env/py3.6/bin/activate
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 python -u $main --mode $mode \
--config /users7/zllei/blackbox-attack/experiments/configs/parsing/stackptr.json \
--seed $seed \
 --num_epochs $epoch --patient_epochs $patient --batch_size $batch --eval_batch_size $evalbatch \
 --opt $opt --schedule $sched --learning_rate $lr --lr_decay $decay --decay_steps $dstep \
 --beta1 $beta1 --beta2 $beta2 --eps $eps --grad_clip $clip --beam $beam \
 --eval_every $evalevery --noscreen ${random_word} ${pretrain_word} $freeze \
 --loss_type $losstype --warmup_steps $warmup --reset $reset --weight_decay $l2decay --unk_replace $unk \
 --word_embedding sskip --word_path $emb --char_embedding random \
 --max_vocab_size ${vocab_size} $trim $ndigit \
 --elmo_path ${elmo_path} ${use_elmo} \
 --pretrained_lm $lm --lm_path $lmpath --lm_lr $lmlr \
 --punctuation '.' '``' "''" ':' ',' --pos_idx $posidx \
 --format $form \
 --train $train \
 --dev $dev \
 --test $test \
    --add_path $add_path \
    --add_flag $add_flag \
    --is_shuffle \
    --total_num $total_num \
    --ratio $ratio \
 --lan_train $lans --lan_dev $lans --lan_test $lans $mix \
 --model_path $save > $log_file