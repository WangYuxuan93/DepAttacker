__author__ = 'max'

import numpy as np
import torch
import random

from neuronlp2.io.common import PAD_TYPE
from neuronlp2.utils import get_oracle_actions
import time

def get_batch(data, batch_size, unk_replace=0.):
    data, data_size = data
    batch_size = min(data_size, batch_size)
    index = torch.randperm(data_size).long()[:batch_size]

    lengths = data['LENGTH'][index]
    max_length = lengths.max().item()
    words = data['WORD']
    single = data['SINGLE']
    words = words[index, :max_length]
    single = single[index, :max_length]
    if unk_replace:
        ones = single.new_ones(batch_size, max_length)
        noise = single.new_empty(batch_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH'] + stack_keys)
    stack_keys = set(stack_keys)
    batch = {'WORD': words, 'LENGTH': lengths}
    batch.update({key: field[index, :max_length] for key, field in data.items() if key not in exclude_keys})
    batch.update({key: field[index, :2 * max_length - 1] for key, field in data.items() if key in stack_keys})
    return batch


def get_bucketed_batch(data, batch_size, unk_replace=0.):
    data_buckets, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    data = data_buckets[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]

    lengths = data['LENGTH'][index]
    max_length = lengths.max().item()
    words = data['WORD']
    single = data['SINGLE']
    words = words[index, :max_length]
    single = single[index, :max_length]
    if unk_replace:
        ones = single.new_ones(batch_size, max_length)
        noise = single.new_empty(batch_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH'] + stack_keys)
    stack_keys = set(stack_keys)
    batch = {'WORD': words, 'LENGTH': lengths}
    batch.update({key: field[index, :max_length] for key, field in data.items() if key not in exclude_keys})
    batch.update({key: field[index, :2 * max_length - 1] for key, field in data.items() if key in stack_keys})
    return batch

def multi_language_iterate_batch(datas, batch_size, unk_replace=0., shuffle=False, switch_lan=False):
    datas, data_sizes = datas
    iterators = []
    for data, data_size in zip(datas, data_sizes):
        lan_id = data['LANG']
        iterators.append({'lan_id':lan_id, 'iter': iterate_batch((data, data_size), batch_size, unk_replace=0., shuffle=False)})
    for it in iterators:
        batch = next(it['iter'], None)
        while batch:
            yield it['lan_id'], batch
            batch = next(it['iter'], None)

def multi_language_iterate_bucketed_batch(datas, batch_size, unk_replace=0., shuffle=False, switch_lan=False):
    datas, data_sizes = datas
    iterators = []
    for data, bucket_sizes in zip(datas, data_sizes):
        lan_id = data[0]['LANG']
        iterators.append({'lan_id':lan_id, 'iter': iterate_bucketed_batch((data, bucket_sizes), batch_size, unk_replace=0., shuffle=False)})
    #print (iterators)
    if switch_lan:
        while iterators:
            cur_idx = random.randint(0, len(iterators)-1)
            it = iterators[cur_idx]
            batch = next(it['iter'], None)
            if batch:
                yield it['lan_id'], batch
            else:
                del iterators[cur_idx]
                #print (iterators)
    else:
        for it in iterators:
            batch = next(it['iter'], None)
            while batch:
                yield it['lan_id'], batch
                batch = next(it['iter'], None)

def multi_language_iterate_data(datas, batch_size, bucketed=False, unk_replace=0., shuffle=False, switch_lan=False):
    if bucketed:
        return multi_language_iterate_bucketed_batch(datas, batch_size, unk_replace=unk_replace, shuffle=shuffle, switch_lan=switch_lan)
    else:
        return multi_language_iterate_batch(datas, batch_size, unk_replace=unk_replace, shuffle=shuffle, switch_lan=switch_lan)


def iterate_batch(data, batch_size, unk_replace=0., shuffle=False):
    data, data_size = data
    words = data['WORD']
    single = data['SINGLE']
    max_length = words.size(1)

    if unk_replace:
        ones = single.new_ones(data_size, max_length)
        noise = single.new_empty(data_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(words.device)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC', 'LANG', 'ERR_TYPE'] + stack_keys)
    stack_keys = set(stack_keys)
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        lengths = data['LENGTH'][excerpt]
        batch_length = lengths.max().item()
        batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths, 'SRC': data['SRC'][excerpt]}
        if 'ERR_TYPE' in data:
            batch['ERR_TYPE'] = data['ERR_TYPE'][excerpt]
        batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
        batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
        yield batch


def iterate_bucketed_batch(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC', 'LANG', 'ERR_TYPE'] + stack_keys)
    stack_keys = set(stack_keys)
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            if 'SRC' in data:
                batch['SRC'] = data['SRC'][excerpt]
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
            batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
            yield batch


def iterate_data(data, batch_size, bucketed=False, unk_replace=0., shuffle=False, task_type="dp",type_alphabet=None,action_alphabet=None,logger=None, **kwargs):
    if task_type == "dp":
        if bucketed:
            return iterate_bucketed_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
        else:
            return iterate_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
    else:
        if kwargs.get("model_type", "") and kwargs.get("model_type")=="SdpPointerParser":  # sdp
            if bucketed:
                return iterate_bucketed_batch_sdp_ptr(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
            else:
                return iterate_batch_sdp_ptr(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
        elif kwargs.get("model_type", "") and kwargs.get("model_type") == "SdpTbParser":
            if bucketed:
                return iterate_bucketed_batch_sdp_tb(data, batch_size, unk_replace=unk_replace, shuffle=shuffle,
                                                     type_alphabet=type_alphabet,action_alphabet=action_alphabet,logger=logger)
            else:
                return iterate_batch_sdp_tb(data, batch_size, unk_replace=unk_replace, shuffle=shuffle,
                                            type_alphabet=type_alphabet,action_alphabet=action_alphabet)
        else:
            if bucketed:
                return iterate_bucketed_batch_sdp(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
            else:
                return iterate_batch_sdp(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)


def iterate_data_dp(data, batch_size, bucketed=False, unk_replace=0., shuffle=False, **kwargs):
    if bucketed:
        return iterate_bucketed_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
    else:
        return iterate_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def iterate_bucketed_batch_sdp(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC','HEAD','TYPE'] + stack_keys)
    sdp_keys = ["HEAD", "TYPE"]
    stack_keys = set(stack_keys)
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths, 'SRC': data['SRC'][excerpt]}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
            batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
            batch.update({key: field[excerpt, :batch_length,:batch_length] for key, field in data.items() if key  in sdp_keys})

            yield batch

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def iterate_batch_sdp(data, batch_size, unk_replace=0., shuffle=False):
    data, data_size = data
    words = data['WORD']
    single = data['SINGLE']
    max_length = words.size(1)

    if unk_replace:
        ones = single.new_ones(data_size, max_length)
        noise = single.new_empty(data_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(words.device)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC', 'HEAD', 'TYPE'] + stack_keys)
    stack_keys = set(stack_keys)
    sdp_keys = ['HEAD', 'TYPE']
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        lengths = data['LENGTH'][excerpt]
        batch_length = lengths.max().item()
        batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths, 'SRC': data['SRC'][excerpt]}
        batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
        batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
        batch.update({key: field[excerpt, :batch_length,:batch_length] for key, field in data.items() if key  in sdp_keys})
        yield batch

# sdp pointer
def iterate_bucketed_batch_sdp_ptr(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    no_truncate = ['LENGTH', 'stacked_heads', 'children', 'siblings', 'stacked_types', 'skip_connect', 'previous', 'next', 'masks_d_t', 'lengths_d_t', 'SRC']
    single_truncate = ['WORD', 'Lemma_s', 'CHAR', 'POS', 'MASK','Lemma_s',"lemmas","SINGLE","PRETRAINED"]
    double_truncate = ['HEAD', 'TYPE']
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch={}
            batch.update({key: field[excerpt] for key, field in data.items() if key in no_truncate})
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key in single_truncate})
            batch.update({key: field[excerpt, :batch_length, :] for key, field in data.items() if key in double_truncate})
            yield batch

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def iterate_batch_sdp_ptr(data, batch_size, unk_replace=0., shuffle=False):
    data, data_size = data
    for start_idx in range(0, len(data)):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt][0]


# transtion-based
def iterate_bucketed_batch_sdp_tb(data, batch_size, unk_replace=0., shuffle=False,type_alphabet=None,action_alphabet=None,logger=None):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    no_truncate = ['LENGTH', 'stacked_heads', 'children', 'siblings', 'stacked_types', 'skip_connect', 'previous', 'next', 'masks_d_t', 'lengths_d_t', 'SRC']
    single_truncate = ['WORD', 'Lemma_s', 'CHAR', 'POS', 'MASK','Lemma_s',"lemmas","SINGLE","PRETRAINED"]
    double_truncate = ['HEAD', 'TYPE']
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)

        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch={}
            # ************ get actions *************************#
            # annotated_sentence, directed_arc_indices, arc_tags
            # get actions:
            tokens_ = data["SRC"][excerpt]
            list_layer = 0
            if isinstance(tokens_,list):
                list_layer += 1
                new_tokens = tokens_[0]
                while isinstance(new_tokens,list):
                    list_layer +=1
                    new_tokens = new_tokens[0]
            if list_layer == 1:
                tokens_ = [tokens_]

            # 只有一个样例  Jeffrey
            # if isinstance(tokens_, np.ndarray):
            #     if not isinstance(tokens_[0],list):
            #         b = tokens_.shape
            #         tokens_ = tokens_.reshape((1, b[0]))
            heads_ = data["HEAD"][excerpt,:,:]
            types_ = data["TYPE"][excerpt,:,:]
            batch_size_,token_num,_= heads_.size()
            gold_actions = []
            try:
                for i in range(batch_size_):
                    tokens = tokens_[i]
                    arc_indices = []
                    arc_tag = []
                    for k in range(token_num):
                        for j in range(token_num):
                            if j==0:continue
                            if heads_[i,j,k] == 1:
                                arc_indices.append((j, k))
                                arc_tag.append(type_alphabet.get_instance(types_[i,j,k]))

                    gold_action = get_oracle_actions(tokens[1:],arc_indices,arc_tag)
                    gold_id = [action_alphabet.get_index(l) for l in gold_action]
                    gold_actions.append(gold_id)

                max_action_num = max([len(x) for x in gold_actions])
                action_tensor = np.empty([batch_size, max_action_num], dtype=np.int32)
                for pre, x in enumerate(gold_actions):
                    action_tensor[pre,0:len(x)] = x
                    action_tensor[pre,len(x):] = action_alphabet.get_index(PAD_TYPE)
                batch.update({key: field[excerpt] for key, field in data.items() if key in no_truncate})
                batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key in single_truncate})
                batch.update({key: field[excerpt, :batch_length, :] for key, field in data.items() if key in double_truncate})
                batch.update({"oracle_actions":gold_actions})
                batch.update({"gold_actions":action_tensor})
            except:
                print("组装数据出错")
                print(tokens_)
                continue
            yield batch

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def iterate_batch_sdp_tb(data, batch_size, unk_replace=0., shuffle=False,action_alphabet=None,type_alphabet=None):
    data, data_size = data
    words = data['WORD']
    single = data['SINGLE']
    max_length = words.size(1)
    if unk_replace:
        ones = single.new_ones(data_size, max_length)
        noise = single.new_empty(data_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(words.device)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC', 'HEAD', 'TYPE'] + stack_keys)
    stack_keys = set(stack_keys)
    sdp_keys = ['HEAD', 'TYPE']
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        lengths = data['LENGTH'][excerpt]
        batch_length = lengths.max().item()
        # 产生新的gold action:
        tokens_ = data['SRC'][excerpt]
        heads_ = data["HEAD"][excerpt, :, :]
        types_ = data["TYPE"][excerpt, :, :]
        batch_size, token_num, _ = heads_.size()
        gold_actions = []
        arc_ind = []
        arc_tags = []
        arc_infos = []
        # ************ 计算 tokens_ list的嵌套层数 *****************8
        list_layer = 0
        if isinstance(tokens_, list):
            list_layer += 1
            new_tokens = tokens_[0]
            while isinstance(new_tokens, list):
                list_layer += 1
                new_tokens = new_tokens[0]
        if list_layer == 1:
            tokens_ = [tokens_]
        # 只有一个样例  Jeffrey
        # if isinstance(tokens_, np.ndarray):
        #     b = tokens_.shape
        #     if len(b)<2:
        #         tokens_ = tokens_.reshape((1, b[0]))
        # ************** End ******************
        # print(type(tokens_))
        # print(tokens_)
        # 计算gold action:
        for i in range(batch_size):
            tokens = tokens_[i]
            arc_indices = []
            arc_tag = []
            arc_info = []
            for k in range(token_num):
                for j in range(token_num):
                    if j == 0: continue
                    if heads_[i, j, k] == 1:
                        arc_indices.append((j, k))
                        arc_tag.append(type_alphabet.get_instance(types_[i, j, k]))
                        arc_info.append((j,k,type_alphabet.get_instance(types_[i, j, k])))  # 用于计算metric
            # print(tokens)
            gold_action = get_oracle_actions(tokens[1:], arc_indices, arc_tag)
            gold_id = [action_alphabet.get_index(l) for l in gold_action]
            gold_actions.append(gold_id)
            arc_ind.append(arc_indices)
            arc_tags.append(arc_tag)
            arc_infos.append(arc_info)
        max_action_num = max([len(x) for x in gold_actions])
        action_tensor = np.empty([batch_size, max_action_num], dtype=np.int32)
        for pre, x in enumerate(gold_actions):
            action_tensor[pre, 0:len(x)] = x
            action_tensor[pre, len(x):] = action_alphabet.get_index(PAD_TYPE)

        batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths, 'SRC': data['SRC'][excerpt]}
        batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
        batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
        batch.update({key: field[excerpt, :batch_length, :batch_length] for key, field in data.items() if key in sdp_keys})
        batch.update({"oracle_actions": gold_actions})
        batch.update({"gold_actions": action_tensor})
        batch.update({"arc_ind": arc_ind})
        batch.update({"arc_tags": arc_tags})
        batch.update({"arc_infos": arc_infos})
        yield batch



if __name__ == '__main__':
    easyfirst_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
    batch = {'WORD':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'MASK':[[1,1,1,1,1,0],[1,1,1,1,1,1]],
             'POS':[[0,1,2,3,4,5],[0,6,7,8,0,0]], 'LENGTH':np.array([6,5]),
             'CHAR':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'HEAD':[[0,3,1,0,3,5],[0,6,7,8,0,0]],'TYPE':[[0,1,2,3,4,5],[0,6,7,8,0,0]]}
    lengths = batch['LENGTH']
    sample_generate_order(batch, lengths, n_recomp=3)
