import argparse
import json
import os

def load_config(filename):
    conf = json.loads(open(filename, 'r').read())
    return conf

class Attacker(object):
    def __init__(self, gpu, config, common_config, output_dir):
        self.tasks = load_config(config)
        self.common_dict = load_config(common_config)
        self.common_dict["gpu"] = gpu
        self.output_dir = output_dir
        # self.src = "source "+self.common_dict["src_path"]+" ; export LANG='en_US.UTF-8'; "
        self.src=""
        #print (self.common_dict)

    def build_cmd(self, dict):
        cmd = "CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=4 python -u {main_path} --mode {mode} --seed {seed} --beam {beam} {ndigit} {ens} \
--generators {generator} --filters {filter} --tagger {tagger} {use_pad} {cached_path} \
--cand_cache_path {cand_cache_path} --train_vocab {train_vocab} \
{wordpiece_backoff} {dynamic_mlm_cand} \
--batch_size {batch} --adv_batch_size {adv_batch} \
--elmo_path {elmo_path} {use_elmo} {use_pretrain} {use_random} \
--pretrained_lm {lm} --lm_path {lm_path} --sent_encoder_path {sent_path} \
--mlm_cand_file {mlm_cand_file} \
--cand_mlm {cand_mlm_path} --temp {temp} --top_k {top_k} --n_mlm_cands {n_mlm_cands} \
--adv_lm_path {adv_lm_path} \
--ensemble \
--vocab {vocab_path} --cand {cand_path} --syn {syn_path} --knn_path {knn_path} \
--min_word_cos_sim {wsim} --min_sent_cos_sim {ssim} --max_knn_candidates {max_knn} \
--adv_rel_ratio {rel_ratio} --adv_fluency_ratio {flu_ratio} \
--max_mod_percent {max_mod_percent} --ppl_inc_thres {ppl_diff_thres} \
--test {test} --model_path {parser} \
--output_filename {orig_output} --adv_filename {adv_output} --adv_gold_filename {adv_gold_output} \
--noscreen --punctuation \'.\' \'``\' \"\'\'\" \':\' \',\' \
--format conllx --lan_test en --pos_idx 3 > {log} 2>&1".format(**dict)
        return cmd

    def prepare_dir(self, name, config, must_no_path=True):
        task_path = os.path.join(self.output_dir, name)
        if os.path.exists(task_path) and must_no_path:
                        print ("\n###################")
                        print ("path {} already exists, quit\n".format(task_path))
                        return None
        elif must_no_path:
            os.mkdir(task_path)
        model_name = config['parser'].rstrip('/').rsplit('/')[-1]
        data_name = self.common_dict['test'].rstrip('/').rsplit('/')[-1]
        output = model_name+'@'+data_name
        config['orig_output'] = os.path.join(task_path, output+'.orig')
        params = ['mode','rel_ratio','flu_ratio','ppl_diff_thres','wsim','ssim','max_mod_percent']
        adv_name = output+'.adv@'+'-'.join([str(config[p]) for p in params])
        config['adv_output'] = os.path.join(task_path, adv_name)
        config['adv_gold_output'] = os.path.join(task_path, adv_name) + '.gold'
        config['log'] = os.path.join(task_path, name+'.log')
        config['cand_cache_path'] = os.path.join(task_path, 'cand_cache.json')
        task_conf = self.common_dict.copy()
        task_conf.update(config)
        #config.update(self.common_dict)
        conf_path = os.path.join(task_path, 'config.json')
        json.dump(task_conf, open(conf_path, 'w'), indent=4)
        return task_conf

    def run(self, name, cmd):
        cmd = self.src + cmd
        print ("\n##### Running task: {} #####".format(name))
        print (cmd)
        os.system(cmd)

    def start(self):
        for name in self.tasks:
            #print (name)
            #print (self.tasks[name])
            task_config = self.prepare_dir(name, self.tasks[name])
            if task_config is not None:
                #print (task_config)
                cmd = self.build_cmd(task_config)
                self.run(name, cmd)

    def sub_vocab_cmd(self, dict):
        cmd = "./scripts/analyze/sub_vocab.sh {adv_gold_output}".format(**dict)
        return cmd

    def get_sub_vocab(self):
        for name in self.tasks:
            task_config = self.prepare_dir(name, self.tasks[name], False)
            cmd = self.sub_vocab_cmd(task_config)
            self.run(name, cmd)

args_parser = argparse.ArgumentParser(description='attack pipeline')
args_parser.add_argument('--gpu', type=str, help='gpu')
args_parser.add_argument('config', type=str, help='experiment config')
args_parser.add_argument('common_config', type=str, help='common config')
args_parser.add_argument('output_dir', type=str, default='attack_output', help='attack output dir')
args_parser.add_argument('--sub_vocab', action='store_true', default=False, help='generate substitute vocabs')
args = args_parser.parse_args()

configs = args.config.split(':')
for config in configs:
    attacker = Attacker(args.gpu, config, args.common_config, args.output_dir)
#print (attacker.build_cmd(attacker.paths))
    if args.sub_vocab:
        attacker.get_sub_vocab()
        exit()
    attacker.start()
