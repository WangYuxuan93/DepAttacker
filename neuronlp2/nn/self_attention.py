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
from neuronlp2.nn.utils import freeze_embedding

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SinPositionalEmbedding(nn.Module):

    def __init__(self, hidden_size, max_position_embeddings=512):
        super(SinPositionalEmbedding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_position_embeddings, hidden_size))

    def _get_sinusoid_encoding_table(self, max_position_embeddings, hidden_size):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / hidden_size) for hid_j in range(hidden_size)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_position_embeddings)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Input:
            x: (batch, seq_len, hidden_size)
        """
        return x + self.pos_table[:, :x.size(1)].detach()


class AttentionEncoderConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                input_size=200,
                hidden_size=200,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=800,
                hidden_act="gelu",
                dropout_type="seq",
                embedding_dropout_prob=0.33,
                hidden_dropout_prob=0.2,
                inter_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                use_input_layer=False,
                use_sin_position_embedding=False,
                freeze_position_embedding=True,
                max_position_embeddings=512,
                initializer="default",
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            dropout_type: ['seq', 'default']
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            inter_dropout_prob: The dropout prob for intermediate layer
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            initializer: ['orthogonal', 'default', 'xavier_normal'] init type for Linear layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.dropout_type = dropout_type
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.inter_dropout_prob = inter_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_input_layer = use_input_layer
        self.use_sin_position_embedding = use_sin_position_embedding
        self.freeze_position_embedding = freeze_position_embedding
        self.max_position_embeddings = max_position_embeddings
        self.initializer = initializer
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
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

def reset_bias_with_orthogonal(bias):
    bias_temp = torch.nn.Parameter(torch.FloatTensor(bias.size()[0], 1))
    nn.init.orthogonal_(bias_temp)
    bias_temp = bias_temp.view(-1)
    bias.data = bias_temp.data

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

class SelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size, bias=False, initializer=config.initializer)
        self.key = Linear(config.hidden_size, self.all_head_size, bias=False, initializer=config.initializer)
        self.value = Linear(config.hidden_size, self.all_head_size, bias=False, initializer=config.initializer)

        self.dropout = Dropout('default', config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # (batch, seq_len, hidden_size) => (batch, seq_len, num_head, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # (batch, seq_len, num_head, head_size) => (batch, num_head, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # input: (batch, seq_len, hidden_size)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # (batch, seq_len, num_head, head_size) => (batch, num_head, seq_len, head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # (batch, num_head, seq_len, seq_len)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # (batch, num_head, seq_len, seq_len) * (batch, num_head, seq_len, head_size) 
        # => (batch, num_head, seq_len, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch, num_head, seq_len, head_size) => (batch, seq_len, num_head, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # (batch, seq_len, num_head, head_size) => (batch, seq_len, hidden_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=False, initializer=config.initializer)
        self.LayerNorm = LayerNorm(config)
        self.dropout = Dropout(config.dropout_type, config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self.self = SelfAttentionLayer(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class IntermediateLayer(nn.Module):
    def __init__(self, config):
        super(IntermediateLayer, self).__init__()
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


class OutputLayer(nn.Module):
    def __init__(self, config):
        super(OutputLayer, self).__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, initializer=config.initializer)
        self.LayerNorm = LayerNorm(config)
        self.dropout = Dropout(config.dropout_type, config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super(AttentionBlock, self).__init__()
        self.attention = AttentionLayer(config)
        self.intermediate = IntermediateLayer(config)
        self.output = OutputLayer(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class AttentionEncoder_(nn.Module):
    def __init__(self, config):
        super(AttentionEncoder_, self).__init__()
        #layer = AttentionBlock(config)
        #self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.layer = nn.ModuleList([AttentionBlock(config) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def position_encoding_init(n_position,d_pos_vec):
    position_enc=np.array([[pos/np.power(10000,2*(j//2)/d_pos_vec) for j in range(d_pos_vec)] if pos!=0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    position_enc[1:,0::2] = np.sin(position_enc[1:,0::2]) #dim=2i
    position_enc[1:,1::2] = np.cos(position_enc[1:,1::2]) #dim =2i+1
    position_enc=torch.from_numpy(position_enc).type(torch.FloatTensor)
    return position_enc

class AttentionEmbeddings(nn.Module):
    def __init__(self, config):
        super(AttentionEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
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
            freeze_embedding(self.position_embeddings)
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


class AttentionEncoder(nn.Module):
    def __init__(self, config: AttentionEncoderConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(AttentionEncoder, self).__init__()
        self.embeddings = AttentionEmbeddings(config)
        self.encoder = AttentionEncoder_(config)
        self.embedding_output = None

    def forward(self, input_tensor, attention_mask=None):
        """
        Input:
            input_tensor: (batch, seq_len, input_size)
            attention_mask: (batch, seq_len)
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
        self.embedding_output = embedding_output
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)

        return all_encoder_layers

    def get_embedding(self):
        return self.embedding_output
