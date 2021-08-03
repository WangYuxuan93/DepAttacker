# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

class GraphAttentionNetworkConfig(object):
    """Configuration class to store the configuration of a `GraphAttentionModel`.
    """
    def __init__(self,
                input_size=200,
                hidden_size=200,
                num_layers=8,
                num_attention_heads=8,
                share_params=False,
                only_value_weight=False,
                intermediate_size=800,
                hidden_act="gelu",
                dropout_type="seq",
                embedding_dropout_prob=0.33,
                hidden_dropout_prob=0.2,
                inter_dropout_prob=0.1,
                attention_probs_dropout_prob=0,
                use_input_layer=False,
                use_sin_position_embedding=False,
                freeze_position_embedding=False,
                max_position_embeddings=512,
                initializer_range=0.02,
                initializer='default',
                encode_arc_type='max',
                rel_dim=100, do_encode_rel=False,
                use_null_att_pos=False):
        """Constructs BertConfig.

        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            use_null_att_pos: Whether to enable null attention position at index-0, so that 
                the model learns to attend to it when there is no head yet 
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.share_params = share_params
        self.num_layers = num_layers
        self.hidden_act = hidden_act
        self.dropout_type = dropout_type
        self.intermediate_size = intermediate_size
        self.embedding_dropout_prob = hidden_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.inter_dropout_prob = inter_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_input_layer = use_input_layer
        self.use_sin_position_embedding = use_sin_position_embedding
        self.freeze_position_embedding = freeze_position_embedding
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.initializer = initializer
        self.encode_arc_type = encode_arc_type
        self.only_value_weight = only_value_weight
        self.rel_dim = rel_dim
        self.do_encode_rel = do_encode_rel
        self.use_null_att_pos = use_null_att_pos

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = GraphAttentionNetworkConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Linear(nn.Module):
    def __init__(self,d_in,d_out,bias=True, initializer='default'):
        # initializer = ['orthogonal', 'default', 'xavier_normal']
        super(Linear,self).__init__()
        self.linear = nn.Linear(d_in,d_out,bias=bias)
        if initializer == 'orthogonal':
            #print ("Initializing attention encoder linear with orthogonal ...")
            nn.init.orthogonal_(self.linear.weight)
            if bias:
                reset_bias_with_orthogonal(self.linear.bias)
        elif initializer == 'xavier_normal':
            #print ("Initializing attention encoder linear with xavier_normal ...")
            nn.init.xavier_normal_(self.linear.weight)
        elif initializer == 'xavier_uniform':
            #print ("Initializing attention encoder linear with xavier_uniform ...")
            nn.init.xavier_uniform_(self.linear.weight)
        elif initializer == 'kaiming_uniform':
            #print ("Initializing attention encoder linear with kaiming_uniform ...")
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        #else:
            #print ("Initializing attention encoder linear by default ...")
        

    def forward(self,x):
        return self.linear(x)


class Dropout(nn.Module):
    def __init__(self, dropout_type='seq', dropout_prob=0.1):
        # dropout_type = ['seq', 'default']
        super(Dropout,self).__init__()
        self.dropout_type = dropout_type
        if self.dropout_type == 'seq':
            self.dropout = torch.nn.Dropout2d(p=dropout_prob)
        else:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        """
        if self.dropout_type == 'seq':
            return self.dropout(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.dropout(x)


def position_encoding_init(n_position,d_pos_vec):
    position_enc=np.array([[pos/np.power(10000,2*(j//2)/d_pos_vec) for j in range(d_pos_vec)] if pos!=0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    position_enc[1:,0::2] = np.sin(position_enc[1:,0::2]) #dim=2i
    position_enc[1:,1::2] = np.cos(position_enc[1:,1::2]) #dim =2i+1
    position_enc=torch.from_numpy(position_enc).type(torch.FloatTensor)
    return position_enc


class AttentionEmbeddings(nn.Module):
    def __init__(self, config):
        super(AttentionEmbeddings, self).__init__()
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.use_input_layer:
            self.input_layer = Linear(config.input_size, config.hidden_size, initializer=config.initializer)
        else:
            self.input_layer = None
        self.use_sin_position_embedding = config.use_sin_position_embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.input_size)
        if config.use_sin_position_embedding:
            #self.position_embeddings = SinPositionalEmbedding(config.hidden_size, config.max_position_embeddings)        
            self.position_embeddings.weight.data = position_encoding_init(config.max_position_embeddings, config.input_size)
        if config.freeze_position_embedding:
            if not config.use_sin_position_embedding:
                print ("### Freeze Position Embedding should use with Sin Position Embedding ###")
                exit()
            #freeze_embedding(self.position_embeddings)
            self.position_embeddings.weight.requires_grad=False
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config)
        self.dropout = Dropout(config.dropout_type, config.embedding_dropout_prob)

    def forward(self, input_tensor):
        """
        input_tensor: (batch, seq_len, input_size)
        """
        seq_length = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        #if self.use_sin_position_embedding:
        #    embeddings = self.position_embeddings(input_tensor)
        #else:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_tensor + position_embeddings
        if self.input_layer is not None:
            embeddings = self.input_layer(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def matmul_rel(x, y, z=None):
    """
    x: (batch_size, num_heads, seq_len, head_size)
    y: (batch_size, num_heads, seq_len, head_size)
    z: (batch_size, num_heads, seq_len, seq_len, head_size)
    """
    sizes = list(x.size())
    seq_len, hidden_size = sizes[-2:]
    #print (seq_len, hidden_size)
    new_sizes = sizes[:-2] + [seq_len] + sizes[-2:]
    if z is not None:
        assert list(z.size()) == new_sizes
    #print (new_sizes)
    x_ = x.unsqueeze(-2).expand(new_sizes)
    y_ = y.unsqueeze(-3).expand(new_sizes)
    #print ("x_:\n", x_)
    #print ("y_:\n", y_)
    if z is not None:
        y_ = y_ + z
        #print ("y_+z:\n", y_)
    out = (x_ * y_).sum(-1).squeeze(-1)

    return out


class GraphAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads):
        super(GraphAttention, self).__init__()
        self.encode_arc_type = config.encode_arc_type
        self.only_value_weight = config.only_value_weight
        self.do_encode_rel = config.do_encode_rel
        self.use_null_att_pos = config.use_null_att_pos
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.value = Linear(config.hidden_size, self.all_head_size, bias=False, initializer=config.initializer)
        self.dropout = Dropout('default', config.attention_probs_dropout_prob)
        if not self.only_value_weight:
            self.query = Linear(config.hidden_size, self.all_head_size, bias=False, initializer=config.initializer)
            self.key = Linear(config.hidden_size, self.all_head_size, bias=False, initializer=config.initializer)
        if self.do_encode_rel:
            self.rel_to_hidden = Linear(config.rel_dim, self.attention_head_size, initializer=config.initializer)

    def transpose_for_scores(self, x):
        # (batch, seq_len, hidden_size) => (batch, seq_len, num_head, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # (batch, seq_len, num_head, head_size) => (batch, num_head, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, graph_matrix, rel_embeddings=None, end_mask=None):
        """
        Inputs:
            input_tensor: (batch, seq_len, hidden_size)
            graph_matrix: (batch, seq_len, seq_len), adjacency matrix
            rel_embeddings: optional (batch, seq_len, seq_len, rel_dim)
            end_mask: (batch, seq_len)
        """
        if self.only_value_weight or self.encode_arc_type == 'soft-copy':
            # input: (batch, seq_len, hidden_size//2)
            value_layer = self.value(input_tensor)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(graph_matrix)
            #smoothing_rate = 0.2
            #attention_probs = attention_probs * (1-smoothing_rate) + 0.5 * smoothing_rate
            # (batch, seq_len, seq_len) * (batch, seq_len, hidden_size//2) 
            # => (batch, seq_len, hidden_size//2)
            context_layer = torch.matmul(attention_probs.float(), value_layer)
        else:
            # input: (batch, seq_len, hidden_size//2)
            mixed_query_layer = self.query(input_tensor)
            mixed_key_layer = self.key(input_tensor)
            mixed_value_layer = self.value(input_tensor)

            # (batch, seq_len, num_head, head_size) => (batch, num_head, seq_len, head_size)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            if self.do_encode_rel and rel_embeddings is not None:
                # (batch, seq_len, seq_len, head_size)
                rel_layer = self.rel_to_hidden(rel_embeddings)
                size = list(rel_layer.size())
                new_size = size[:1] + [self.num_attention_heads] + size[1:]
                # (batch, num_head, seq_len, seq_len, head_size)
                rel_layer = rel_layer.unsqueeze(1).expand(new_size)
                # (batch, num_head, seq_len, seq_len)
                attention_scores = matmul_rel(query_layer, key_layer, rel_layer)
            # Take the dot product between "query" and "key" to get the raw attention scores.
            else:
                # (batch, num_head, seq_len, seq_len)
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if not self.encode_arc_type == 'soft':
                if self.use_null_att_pos:
                    graph_matrix = graph_matrix + end_mask.unsqueeze(1)
                    #print ("aug matrix:\n", graph_matrix)
                # (batch, seq_len, seq_len), mask out non-ajacency relations
                neg_inf_mask = (1-graph_matrix) * -1e9
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + neg_inf_mask.unsqueeze(1)


            # Normalize the attention scores to probabilities.
            # (batch, num_head, seq_len, seq_len)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            #print ("graph_matrix:\n",graph_matrix)
            #print ("attention_scores:\n", attention_scores)
            #print ("attention_probs:\n", attention_probs)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)
            # (batch, num_head, seq_len, seq_len) * (batch, num_head, seq_len, head_size) 
            # => (batch, num_head, seq_len, head_size)
            context_layer = torch.matmul(attention_probs.float(), value_layer)
            # (batch, num_head, seq_len, head_size) => (batch, seq_len, num_head, head_size)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            # (batch, seq_len, num_head, head_size) => (batch, seq_len, hidden_size)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class GATSelfOutput(nn.Module):
    def __init__(self, config):
        super(GATSelfOutput, self).__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, initializer=config.initializer)
        self.LayerNorm = LayerNorm(config)
        self.dropout = Dropout(config.dropout_type, config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GATIntermediate(nn.Module):
    def __init__(self, config):
        super(GATIntermediate, self).__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, initializer=config.initializer)
        if config.hidden_act == 'gelu':
            self.intermediate_act_fn = gelu
        elif config.hidden_act == 'relu':
            self.intermediate_act_fn = nn.ReLU()
        elif config.hidden_act == 'leaky_relu':
            self.intermediate_act_fn = nn.LeakyReLU(0.1)
        self.dropout = Dropout(config.dropout_type, config.inter_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GATOutput(nn.Module):
    def __init__(self, config):
        super(GATOutput, self).__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, initializer=config.initializer)
        self.LayerNorm = LayerNorm(config)
        self.dropout = Dropout(config.dropout_type, config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.input_encoder = None
        # information flow in bidirection
        self.fw_graph_attention = GraphAttention(config, config.hidden_size//2, config.num_attention_heads)
        self.bw_graph_attention = GraphAttention(config, config.hidden_size//2, config.num_attention_heads)
        self.self_output = GATSelfOutput(config)
        self.intermediate = GATIntermediate(config)
        self.output = GATOutput(config)

    def forward(self, hidden_states, graph_matrix, attention_mask, rel_embeddings=None, 
                end_mask=None):
        # (batch, seq_len, hidden_size/2)
        fw_ga_output = self.fw_graph_attention(hidden_states, graph_matrix, rel_embeddings=rel_embeddings, 
                                                end_mask=end_mask)
        if rel_embeddings is not None:
            bw_rel_embeddings = rel_embeddings.permute(0,2,1,3)
        else:
            bw_rel_embeddings = None
        bw_ga_output = self.bw_graph_attention(hidden_states, graph_matrix.transpose(-1,-2), 
                                            rel_embeddings=bw_rel_embeddings, end_mask=end_mask)
        #print ("graph_matrix:\n", graph_matrix)
        # (batch, seq_len, hidden_size)
        concat_output = torch.cat([fw_ga_output,fw_ga_output], dim=-1)
        #print ("fw_ga_output:\n", fw_ga_output)
        #print ("graph_attention_output:\n",graph_attention_output)
        # (batch, seq_len, hidden_size), residual layer
        attention_output = self.self_output(hidden_states, concat_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class GATEncoder(nn.Module):
    def __init__(self, config):
        super(GATEncoder, self).__init__()
        self.share_params = config.share_params
        self.num_layers = config.num_layers
        #layer = GATLayer(config)
        if self.share_params:
            #self.layer = layer
            self.layer = GATLayer(config)
        else:
            #self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_layers)])
            self.layer = nn.ModuleList([GATLayer(config) for _ in range(config.num_layers)])

    def forward(self, hidden_states, graph_matrix, attention_mask, rel_embeddings=None,
                end_mask=None):
        all_encoder_layers = []
        if self.share_params:
            for _ in range(self.num_layers):
                hidden_states = self.layer(hidden_states, graph_matrix, attention_mask, 
                                    rel_embeddings=rel_embeddings, end_mask=end_mask)
                all_encoder_layers.append(hidden_states)
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, graph_matrix, attention_mask, 
                                    rel_embeddings=rel_embeddings, end_mask=end_mask)
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class GraphAttentionNetwork(nn.Module):
    def __init__(self, config: GraphAttentionNetworkConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(GraphAttentionNetwork, self).__init__()
        self.embeddings = AttentionEmbeddings(config)
        self.encoder = GATEncoder(config)

    def forward(self, input_tensor, graph_matrix, attention_mask=None, rel_embeddings=None,
                end_mask=None):
        """
        Input:
            input_tensor: (batch, seq_len, input_size)
            graph_matrix: (batch, seq_len, seq_len)
            attention_mask: (batch, seq_len)
            rel_embeddings: (batch, seq_len, seq_len, rel_dim)
            end_mask: (batch, seq_len)
        """
        if attention_mask is None:
            #attention_mask = torch.ones_like(input_ids)
            # (batch, seq_len)
            attention_mask = torch.ones(input_tensor.size(0),input_tensor.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_tensor)
        all_encoder_layers = self.encoder(embedding_output, graph_matrix, extended_attention_mask,
                                        rel_embeddings=rel_embeddings, end_mask=end_mask)

        return all_encoder_layers
