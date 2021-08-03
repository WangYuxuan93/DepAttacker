#!/usr/bin/env bash
main=/users7/zllei/NeuroNLP2-yx/adversary/succ_rate.py


orig_path=/users7/zllei/exp_data/models/adv/ptb/2021/stackptr/stackptr-glove-v400k-v2-black-ptb_test-0.15-v0/glove-777@PTB_test_auto.conll.orig
adv_path=/users7/zllei/exp_data/models/adv/ptb/2021/transferability/cross-parser/biaf-attack-stackptr-glove-777.conll
gold_path=/users2/yxwang/work/data/ptb/dependency-stanford-chen14/PTB_test_auto.conll
python -u $main --orig $orig_path --adv $adv_path --gold $gold_path --p >&1