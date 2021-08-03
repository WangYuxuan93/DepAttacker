__author__ = 'max'

import numpy as np
import torch


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
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC'] + stack_keys)
    stack_keys = set(stack_keys)
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
        yield batch


def iterate_bucketed_batch(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'SRC'] + stack_keys)
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
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}#, 'SRC': data['SRC'][excerpt]}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
            batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
            yield batch


def iterate_bucketed_batch_and_sample(data, batch_size, step_batch_size=None, unk_replace=0., shuffle=False, max_layers=4):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH'] + stack_keys)
    easyfirst_keys = ['WORD', 'MASK', 'POS', 'CHAR', 'HEAD', 'TYPE']
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
            # [batch_size, batch_len]
            heads = data['HEAD'][excerpt, :batch_length]
            types = data['TYPE'][excerpt, :batch_length]
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key in easyfirst_keys})
            batch_by_layer = sample_generate_order(batch, lengths, n_recomp=max_layers-1)
            if step_batch_size is not None:
                yield split_batch_by_layer(batch_by_layer, step_batch_size, shuffle=False)
            else:
                yield batch_by_layer


def sample_generate_order(batch, lengths, n_recomp=3, recomp_in_prev=False, debug=False):

    RECOMP = -1
    EOS = -2
    NO_RECOMP = 0
    DO_RECOMP = 1
    DO_EOS = 2
    batch_length = lengths.max().item()

    easyfirst_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
    all_keys = easyfirst_keys + ['RECOMP', 'GEN_HEAD', 'NEXT_HEAD']
    batch_by_layer = {i: {key: [] for key in all_keys} for i in range(n_recomp+1)}

    # for every sentence
    for i in range(len(lengths)):
        if recomp_in_prev:
            new_order = np.arange(1,batch_length-1).astype(int)
            np.random.shuffle(new_order)
            new_order = np.append(np.ones(n_recomp).astype(int) * RECOMP, new_order)
            new_order = np.append(new_order, EOS)
        else:
            new_order = np.append(np.arange(1,batch_length-1), np.ones(n_recomp) * RECOMP).astype(int)
            np.random.shuffle(new_order)
            new_order = np.append(new_order, EOS)
        if debug:
            print ("new_order", new_order)
        n_step = 0
        generated_heads = np.zeros([1,batch_length], dtype=np.int32)
        zero_mask = np.zeros([batch_length], dtype=np.int32)
        # the input generated head list
        generated_heads_list = []
        # whether to recompute at this step
        recomp_list = []
        # the next head to be generated, in shape of 0-1 mask
        next_list = []
        while n_step < len(new_order):
            next_step = new_order[n_step]
            if next_step == RECOMP:
                next_list.append(np.copy(zero_mask))
                generated_heads_list.append(np.copy(generated_heads))
                recomp_list.append(DO_RECOMP)
                prev_layers = generated_heads_list[-1]
                # add a new layer
                generated_heads = np.concatenate([prev_layers,np.zeros([1,batch_length], dtype=int)], axis=0)
            elif next_step == EOS:
                next_list.append(np.copy(zero_mask))
                generated_heads_list.append(np.copy(generated_heads))
                recomp_list.append(DO_EOS)
            else:
                next_list.append(np.copy(zero_mask))
                next_list[-1][next_step] = 1
                generated_heads_list.append(np.copy(generated_heads))
                recomp_list.append(NO_RECOMP)
                # add one new head to the top layer of generated heads
                generated_heads[-1,next_step] = 1
            n_step += 1
        """
        print ('generated_heads_list:')
        for h in generated_heads_list:
            print (h)
        print ('recomp_list:', recomp_list)"""
        
        for n_step in range(len(recomp_list)):
            n_layers = len(generated_heads_list[n_step]) - 1
            for key in easyfirst_keys:
                batch_by_layer[n_layers][key].append(batch[key][i])
            batch_by_layer[n_layers]['RECOMP'].append(recomp_list[n_step])
            batch_by_layer[n_layers]['GEN_HEAD'].append(generated_heads_list[n_step])
            batch_by_layer[n_layers]['NEXT_HEAD'].append(next_list[n_step])
    for n_layers in batch_by_layer.keys():
        for key in batch_by_layer[n_layers].keys():
            batch_by_layer[n_layers][key] = torch.from_numpy(np.stack(batch_by_layer[n_layers][key]))
        # (batch, n_layers, seq_len) -> (n_layers, batch, seq_len)
        batch_by_layer[n_layers]['GEN_HEAD'] = np.transpose(batch_by_layer[n_layers]['GEN_HEAD'], (1,0,2))
        
    if debug:
        for i in batch_by_layer.keys():
            print('-' * 50)
            print ("layer-%d"%i)
            for key in batch_by_layer[i].keys():
                print ("%s\n"%key, batch_by_layer[i][key])
    return batch_by_layer
        

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def iterate_data(data, batch_size, bucketed=False, unk_replace=0., shuffle=False, task_type='dp'):
    if bucketed:
        if task_type == 'sdp':
            return iterate_bucketed_batch_sdp(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
        else:
            return iterate_bucketed_batch(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
    else:
        if task_type == 'sdp':
            return iterate_batch_sdp(data, batch_size, unk_replace=unk_replace, shuffle=shuffle)
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
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}#, 'SRC': data['SRC'][excerpt]}
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

def iterate_data_and_sample(data, batch_size, step_batch_size=None, bucketed=False, unk_replace=0., shuffle=False, 
                            max_layers=6):
    return iterate_bucketed_batch_and_sample(data, batch_size, step_batch_size=step_batch_size,
                                unk_replace=unk_replace, shuffle=shuffle, max_layers=max_layers)


def sample_from_model(network, data, batch_size, step_batch_size=None, bucketed=False, 
                      unk_replace=0., shuffle=False, max_layers=6, use_whole_seq=True, 
                      device=torch.device('cpu'), debug=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH'] + stack_keys)
    easyfirst_keys = ['WORD', 'MASK', 'POS', 'CHAR', 'HEAD', 'TYPE']
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
            # [batch_size, batch_len]
            heads = data['HEAD'][excerpt, :batch_length]
            types = data['TYPE'][excerpt, :batch_length]
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key in easyfirst_keys})
            # pre-process the input
            input_word = batch['WORD']#.to(device)
            input_char = batch['CHAR']#.to(device)
            input_pos = batch['POS']#.to(device)
            gold_heads = batch['HEAD'].to(device)
            mask = batch['MASK'].to(device)
            batch_by_layer = network.inference(input_word, input_char, input_pos, gold_heads, 
                                batch, mask=mask, max_layers=max_layers, use_whole_seq=use_whole_seq,
                                device=device)

            if debug:
                for i in batch_by_layer.keys():
                    print('-' * 50)
                    print ("layer-%d"%i)
                    for key in batch_by_layer[i].keys():
                        print ("%s\n"%key, batch_by_layer[i][key])
            if step_batch_size is not None:
                yield split_batch_by_layer(batch_by_layer, step_batch_size, shuffle=False)
            else:
                yield batch_by_layer

def split_batch_by_layer(batch_by_layer, step_batch_size, shuffle=False, debug=False):

    batches = []
    keys_ = batch_by_layer[0].keys() - ['LENGTH', 'RECOMP', 'GEN_HEAD']
    for n_layers in batch_by_layer.keys():
        if batch_by_layer[n_layers]['LENGTH'] is None: continue
        bucket_size = len(batch_by_layer[n_layers]['LENGTH'])
        for start_idx in range(0, bucket_size, step_batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + step_batch_size]
            else:
                excerpt = slice(start_idx, start_idx + step_batch_size)
            # [step_batch_size, batch_len]
            batch = {'LENGTH': batch_by_layer[n_layers]['LENGTH'][excerpt],
                     'RECOMP': batch_by_layer[n_layers]['RECOMP'][excerpt],
                     'GEN_HEAD': batch_by_layer[n_layers]['GEN_HEAD'][:,excerpt,:]}
            batch.update({key: field[excerpt, :] for key, field in batch_by_layer[n_layers].items() if key in keys_})
            batches.append(batch)

    if debug:
        print ("Split batches:")
        for batch in batches:
            print('-' * 50)
            for key in batch.keys():
                print ("%s\n"%key, batch[key])

    return batches


def get_order_mask(lengths, sampler='random',seed=None, symbolic_end=True):
    batch_length = max(lengths)
    order_masks = []
    if sampler == 'random':
        for i in range(len(lengths)):
            seq_len = lengths[i]
            if symbolic_end:
                new_order = np.arange(1,seq_len-1)
            else:
                new_order = np.arange(1,seq_len)
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(new_order)
            order_mask = np.eye(batch_length)[new_order]
            order_mask = np.pad(order_mask, ((0,batch_length-seq_len),(0,0)), 'constant', constant_values=(0,0))
            #print (order_mask)
            order_masks.append(np.expand_dims(order_mask, axis=1))
        # (seq_len, batch_size, seq_len)
        order_masks = torch.from_numpy(np.concatenate(order_masks, axis=1))#.permute(1,0,2)
        #print (order_masks)
    elif sampler == 'from_model':
        print ("Not Implemented")

    return order_masks

if __name__ == '__main__':
    easyfirst_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
    batch = {'WORD':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'MASK':[[1,1,1,1,1,0],[1,1,1,1,1,1]],
             'POS':[[0,1,2,3,4,5],[0,6,7,8,0,0]], 'LENGTH':np.array([6,5]),
             'CHAR':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'HEAD':[[0,3,1,0,3,5],[0,6,7,8,0,0]],'TYPE':[[0,1,2,3,4,5],[0,6,7,8,0,0]]}
    lengths = batch['LENGTH']
    sample_generate_order(batch, lengths, n_recomp=3)
