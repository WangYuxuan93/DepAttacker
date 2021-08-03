from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.io import get_logger
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN, BiAffine_v2
from neuronlp2.tasks import parser
from neuronlp2.nn.self_attention import AttentionEncoderConfig, AttentionEncoder
from neuronlp2.nn.graph_attention_network import GraphAttentionNetworkConfig, GraphAttentionNetwork
from neuronlp2.nn.dropout import drop_input_independent
from torch.autograd import Variable
from transformers import *
from neuronlp2.models.biaffine_parser import BiaffineParser
import random

class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2

class StackPointerParser(BiaffineParser):
    def __init__(self, hyps, num_pretrained, num_words, num_chars, num_pos, num_labels,
                 device=torch.device('cpu'),
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 use_pretrained_static=True, use_random_static=False,
                 use_elmo=False, elmo_path=None, 
                 pretrained_lm='none', lm_path=None, num_lans=1):
        self.dec_out_dim = hyps['decoder']['hidden_size']
        super(StackPointerParser, self).__init__(hyps, num_pretrained, num_words, num_chars, num_pos,
                               num_labels, device=device,
                               embedd_word=embedd_word, embedd_char=embedd_char, 
                               use_pretrained_static=use_pretrained_static, 
                               use_random_static=use_random_static,
                               use_elmo=use_elmo, elmo_path=elmo_path,
                               pretrained_lm=pretrained_lm, lm_path=lm_path,
                               num_lans=num_lans)

        prior_order = hyps['input']['prior_order']
        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)

        self.grandPar = hyps['input']['grandPar']
        self.sibling = hyps['input']['sibling']
        logger = get_logger("StackPtr")
        logger.info("##### StackPtr Parser (PriorOrder: %s, GrandPar: %s, Sibling: %s) #####" % (prior_order, self.grandPar, self.sibling))

        # for decoder
        decoder_name = hyps['decoder']['name']
        decoder_hidden_size = hyps['decoder']['hidden_size']
        decoder_layers = hyps['decoder']['num_layers']
        encoder_hidden_size = hyps['input_encoder']['hidden_size']
        
        logger.info("##### Decoder (Type: %s, Layer: %d, Hidden: %d) #####" % (decoder_name, decoder_layers, decoder_hidden_size))
        if decoder_name == 'RNN':
            RNN_DECODER = VarRNN
        elif decoder_name == 'LSTM':
            RNN_DECODER = VarLSTM
        elif decoder_name == 'FastLSTM':
            RNN_DECODER = VarFastLSTM
        elif decoder_name == 'GRU':
            RNN_DECODER = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % decoder_name)

        dec_dim = encoder_hidden_size // 2 # 维度减半
        self.src_dense = nn.Linear(self.enc_out_dim, dec_dim)  # for decode
        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(dec_dim, decoder_hidden_size, num_layers=decoder_layers, batch_first=True, bidirectional=False, dropout=self.p_rnn)

        self.hx_dense = nn.Linear(2 * decoder_hidden_size, decoder_hidden_size) # 用于构造解码的初始h0

    def init_biaffine(self):
        hid_size = self.dec_out_dim
        self.arc_h = nn.Linear(self.dec_out_dim, self.arc_mlp_dim)  # decode 出来的是head的表示
        self.arc_c = nn.Linear(self.enc_out_dim, self.arc_mlp_dim)  # encode 出来的是argument的表示
        self.arc_attention = BiAffine(self.arc_mlp_dim, self.arc_mlp_dim)
        self.basic_parameters.append(self.arc_h)
        self.basic_parameters.append(self.arc_c)
        self.basic_parameters.append(self.arc_attention)

        self.rel_h = nn.Linear(self.dec_out_dim, self.rel_mlp_dim)
        self.rel_c = nn.Linear(self.enc_out_dim, self.rel_mlp_dim)
        self.rel_attention = BiLinear(self.rel_mlp_dim, self.rel_mlp_dim, self.num_labels)
        self.basic_parameters.append(self.rel_h)
        self.basic_parameters.append(self.rel_c)
        self.basic_parameters.append(self.rel_attention)

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask=None):
        # get vector for heads [batch, length_decoder, input_dim],
        enc_dim = output_enc.size(2)
        batch, length_dec = heads_stack.size()
        src_encoding = output_enc.gather(dim=1, index=heads_stack.unsqueeze(2).expand(batch, length_dec, enc_dim))
        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sib = siblings.gt(0).float().unsqueeze(2)
            output_enc_sibling = output_enc.gather(dim=1, index=siblings.unsqueeze(2).expand(batch, length_dec, enc_dim)) * mask_sib
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [batch, length_decoder, 1]
            gpars = heads.gather(dim=1, index=heads_stack).unsqueeze(2)
            # mask_gpar = gpars.ge(0).float()
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc.gather(dim=1, index=gpars.expand(batch, length_dec, enc_dim)) #* mask_gpar
            src_encoding = src_encoding + output_enc_gpar
        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = self.activation(self.src_dense(src_encoding))
        #print ("src_encoding:\n", src_encoding)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, mask, hx=hx)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
        return hn

    def forward(self, input_word, input_pretrained, input_char, input_pos, heads, stacked_heads, 
                children, siblings, stacked_rels, mask_e=None, mask_d=None, 
                bpes=None, first_idx=None, input_elmo=None, lan_id=None):
        # (batch, seq_len, embed_size)
        embeddings = self._embed(input_word, input_pretrained, input_char, input_pos, 
                                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id)
        # (batch, seq_len, hidden_size)
        output_enc, hn = self._input_encoder(embeddings, mask=mask_e, lan_id=lan_id)
        #print ("output_enc:\n", output_enc)
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        rel_c = self.activation(self.rel_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        output_dec, _ = self._get_decoder_output(output_enc, heads, stacked_heads, siblings, hn, mask=mask_d)
        #print ("output_dec:\n", output_dec)
        # output size [batch, length_decoder, arc_space]
        arc_h = self.activation(self.arc_h(output_dec))
        rel_h = self.activation(self.rel_h(output_dec))

        batch, max_len_d, rel_space = rel_h.size()

        # apply dropout
        # [batch, length_decoder, dim] + [batch, length_encoder, dim] --> [batch, length_decoder + length_encoder, dim]
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        rel = self.dropout_out(torch.cat([rel_h, rel_c], dim=1).transpose(1, 2)).transpose(1, 2)
        rel_h = rel[:, :max_len_d].contiguous()
        rel_c = rel[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        arc_logits = self.arc_attention(arc_h, arc_c, mask_query=mask_d, mask_key=mask_e)
        # get vector for heads [batch, length_decoder, rel_space],
        rel_c = rel_c.gather(dim=1, index=children.unsqueeze(2).expand(batch, max_len_d, rel_space))
        # compute output for type [batch, length_decoder, num_labels]
        rel_logits = self.rel_attention(rel_h, rel_c)
        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_mask_e = mask_e.eq(0).unsqueeze(1)
            minus_mask_d = mask_d.eq(0).unsqueeze(2)
            arc_logits = arc_logits.masked_fill(minus_mask_d * minus_mask_e, float('-inf'))

        # loss_arc shape [batch, length_decoder]
        loss_arc = self.criterion(arc_logits.transpose(1, 2), children)
        loss_rel = self.criterion(rel_logits.transpose(1, 2), stacked_rels)

        if mask_d is not None:
            loss_arc = loss_arc * mask_d
            loss_rel = loss_rel * mask_d

        return (loss_arc.sum(dim=1), loss_rel.sum(dim=1))

    def decode(self, input_word, input_pretrained, input_char, input_pos, mask=None, beam=1, 
                bpes=None, first_idx=None, input_elmo=None, lan_id=None, leading_symbolic=0):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output_enc [batch, length, model_dim]
        # arc_c [batch, length, arc_space]
        # rel_c [batch, length, rel_space]
        # hn [num_direction, batch, hidden_size]
        # (batch, seq_len, embed_size)
        # Jeffrey: 先进行embedding
        embeddings = self._embed(input_word, input_pretrained, input_char, input_pos, 
                                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id)
        # (batch, seq_len, hidden_size)
        # Jeffrey: 进行encoder
        output_enc, hn = self._input_encoder(embeddings, mask=mask, lan_id=lan_id)
        # Jeffrey: output_enc hidden的输出， hn为cell state
        enc_dim = output_enc.size(2)
        device = output_enc.device
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, rel_space]
        rel_c = self.activation(self.rel_c(output_enc))
        rel_space = rel_c.size(2)
        # [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()

        heads = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
        rels = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)

        num_steps = 2 * max_len - 1
        stacked_heads = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64)
        siblings = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64) if self.sibling else None
        hypothesis_scores = output_enc.new_zeros((batch, 1))

        # [batch, beam, length]
        children = torch.arange(max_len, device=device, dtype=torch.int64).view(1, 1, max_len).expand(batch, beam, max_len)
        constraints = torch.zeros(batch, 1, max_len, device=device, dtype=torch.bool)
        constraints[:, :, 0] = True
        # [batch, 1]
        batch_index = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1)

        # compute lengths
        if mask is None:
            steps = torch.new_tensor([num_steps] * batch, dtype=torch.int64, device=device)
            mask_sent = torch.ones(batch, 1, max_len, dtype=torch.bool, device=device)
        else:
            steps = (mask.sum(dim=1) * 2 - 1).long()
            mask_sent = mask.unsqueeze(1).bool()

        num_hyp = 1
        mask_hyp = torch.ones(batch, 1, device=device)
        hx = hn  # hx: context
        for t in range(num_steps):
            # [batch, num_hyp]
            curr_heads = stacked_heads[:, :, t]  # 当前head
            curr_gpars = heads.gather(dim=2, index=curr_heads.unsqueeze(2)).squeeze(2) # 查找父节点
            curr_sibs = siblings[:, :, t] if self.sibling else None # 一个兄弟节点
            # [batch, num_hyp, enc_dim]
            src_encoding = output_enc.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim)) # 取出num_hyp个词的嵌入表示

            if self.sibling:
                mask_sib = curr_sibs.gt(0).float().unsqueeze(2)
                output_enc_sibling = output_enc.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [batch, num_hyp, dec_dim]
            src_encoding = self.activation(self.src_dense(src_encoding)) # 进行线性变换，维度压缩

            # output [batch * num_hyp, dec_dim]
            # hx [decoder_layer, batch * num_hyp, dec_dim]
            output_dec, hx = self.decoder.step(src_encoding.view(batch * num_hyp, -1), hx=hx)
            dec_dim = output_dec.size(1)
            # [batch, num_hyp, dec_dim]
            output_dec = output_dec.view(batch, num_hyp, dec_dim)

            # [batch, num_hyp, arc_space]
            arc_h = self.activation(self.arc_h(output_dec))
            # [batch, num_hyp, rel_space]
            rel_h = self.activation(self.rel_h(output_dec))
            # [batch, num_hyp, length]
            arc_logits = self.arc_attention(arc_h, arc_c, mask_query=mask_hyp, mask_key=mask)
            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_mask_enc = mask.eq(0).unsqueeze(1)
                arc_logits.masked_fill_(minus_mask_enc, float('-inf'))

            # [batch]
            mask_last = steps.le(t + 1)
            mask_stop = steps.le(t)
            minus_mask_hyp = mask_hyp.eq(0).unsqueeze(2)
            # [batch, num_hyp, length]
            hyp_scores = F.log_softmax(arc_logits, dim=2).masked_fill_(mask_stop.view(batch, 1, 1) + minus_mask_hyp, 0)
            # [batch, num_hyp, length]
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + hyp_scores

            # [batch, num_hyp, length]
            mask_leaf = curr_heads.unsqueeze(2).eq(children[:, :num_hyp]) * mask_sent
            mask_non_leaf = (~mask_leaf) * mask_sent

            # apply constrains to select valid hyps
            # [batch, num_hyp, length]
            mask_leaf = mask_leaf * (mask_last.unsqueeze(1) + curr_heads.ne(0)).unsqueeze(2)
            mask_non_leaf = mask_non_leaf * (~constraints)

            hypothesis_scores.masked_fill_(~(mask_non_leaf + mask_leaf), float('-inf'))
            # [batch, num_hyp * length]
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch, -1), dim=1, descending=True)
            # 预测的head结果

            # [batch]
            prev_num_hyp = num_hyp
            num_hyps = (mask_leaf + mask_non_leaf).long().view(batch, -1).sum(dim=1)
            num_hyp = num_hyps.max().clamp(max=beam).item()
            # [batch, hum_hyp]
            hyps = torch.arange(num_hyp, device=device, dtype=torch.int64).view(1, num_hyp)
            mask_hyp = hyps.lt(num_hyps.unsqueeze(1)).float()

            # [batch, num_hyp]
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            hyp_index = hyp_index[:, :num_hyp]
            base_index = hyp_index // max_len
            child_index = hyp_index % max_len

            # [batch, num_hyp]
            hyp_heads = curr_heads.gather(dim=1, index=base_index)
            hyp_gpars = curr_gpars.gather(dim=1, index=base_index)

            # [batch, num_hyp, length]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, max_len)
            constraints = constraints.gather(dim=1, index=base_index_expand)
            constraints.scatter_(2, child_index.unsqueeze(2), True)

            # [batch, num_hyp]
            mask_leaf = hyp_heads.eq(child_index)
            # [batch, num_hyp, length]
            heads = heads.gather(dim=1, index=base_index_expand)
            heads.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, hyp_gpars, hyp_heads).unsqueeze(2))
            rels = rels.gather(dim=1, index=base_index_expand)
            # [batch, num_hyp]
            org_rels = rels.gather(dim=2, index=child_index.unsqueeze(2)).squeeze(2)

            # [batch, num_hyp, num_steps]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, num_steps + 1)
            stacked_heads = stacked_heads.gather(dim=1, index=base_index_expand)
            stacked_heads[:, :, t + 1] = torch.where(mask_leaf, hyp_gpars, child_index)
            if self.sibling:
                siblings = siblings.gather(dim=1, index=base_index_expand)
                siblings[:, :, t + 1] = torch.where(mask_leaf, child_index, torch.zeros_like(child_index))

            # [batch, num_hyp, rel_space]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, rel_space)
            child_index_expand = child_index.unsqueeze(2).expand(batch, num_hyp, rel_space)
            # [batch, num_hyp, num_labels]
            rel_logits = self.rel_attention(rel_h.gather(dim=1, index=base_index_expand), rel_c.gather(dim=1, index=child_index_expand))
            hyp_type_scores = F.log_softmax(rel_logits, dim=2)
            # compute the prediction of rels [batch, num_hyp]
            hyp_type_scores, hyp_rels = hyp_type_scores.max(dim=2)
            hypothesis_scores = hypothesis_scores + hyp_type_scores.masked_fill_(mask_stop.view(batch, 1), 0)
            rels.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, org_rels, hyp_rels).unsqueeze(2))

            # hx [decoder_layer, batch * num_hyp, dec_dim]
            # hack to handle LSTM
            hx_index = (base_index + batch_index * prev_num_hyp).view(batch * num_hyp)
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, hx_index]
                cx = cx[:, hx_index]
                hx = (hx, cx)
            else:
                hx = hx[:, hx_index]

        heads = heads[:, 0].cpu().numpy()
        rels = rels[:, 0].cpu().numpy()
        return heads, rels