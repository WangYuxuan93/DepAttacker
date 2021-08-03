# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

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

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
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
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
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

class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

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

class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

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

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class GraphAttentionConfig(object):
    """Configuration class to store the configuration of a `GraphAttentionModel`.
    """
    def __init__(self,
                input_size=100,
                hidden_size=768,
                arc_space=600,
                num_graph_attention_layers=1,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                graph_attention_probs_dropout_prob=0,
                use_input_layer=False,
                use_sin_position_embedding=False,
                max_position_embeddings=512,
                initializer_range=0.02,
                share_params=False,
                only_value_weight=False,
                extra_self_attention_layer=False,
                input_self_attention_layer=False,
                num_input_attention_layers=3):
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
            extra_self_attention_layer: whether to use a BERT self attention layer on top
                of graph attention layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.arc_space = arc_space
        self.num_attention_heads = num_attention_heads
        self.num_graph_attention_layers = num_graph_attention_layers
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.embedding_dropout_prob = hidden_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.graph_attention_probs_dropout_prob = graph_attention_probs_dropout_prob
        self.use_input_layer = use_input_layer
        self.use_sin_position_embedding = use_sin_position_embedding
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.share_params = share_params
        self.only_value_weight = only_value_weight
        self.extra_self_attention_layer = extra_self_attention_layer
        self.input_self_attention_layer = input_self_attention_layer
        self.num_input_attention_layers = num_input_attention_layers

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


class GraphAttentionEmbeddings(nn.Module):
    def __init__(self, config):
        super(GraphAttentionEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.use_input_layer:
            self.input_layer = nn.Linear(config.input_size, config.hidden_size)
        else:
            self.input_layer = None
        self.use_sin_position_embedding = config.use_sin_position_embedding
        if config.use_sin_position_embedding:
            self.position_embeddings = SinPositionalEmbedding(config.hidden_size, config.max_position_embeddings)
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.embedding_dropout_prob)

    def forward(self, input_tensor):
        """
        input_tensor: (batch, seq_len, input_size)
        """
        seq_length = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        if self.input_layer is not None:
            input_tensor = self.input_layer(input_tensor)
        if self.use_sin_position_embedding:
            embeddings = self.position_embeddings(input_tensor)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = input_tensor + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GraphAttention(nn.Module):
    def __init__(self, config):
        super(GraphAttention, self).__init__()

        #self.arc_h = nn.Linear(hidden_size, arc_space)
        #self.arc_c = nn.Linear(hidden_size, arc_space)
        #self.biaffine = BiAffine(arc_space, arc_space)
        self.value = nn.Linear(config.hidden_size, config.arc_space)

        self.dropout = nn.Dropout(config.graph_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # (batch, seq_len, hidden_size) => (batch, seq_len, num_head, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # (batch, seq_len, num_head, head_size) => (batch, num_head, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, graph_matrix):
        """
        Inputs:
            hidden_states: (batch, seq_len, hidden_size)
            graph_matrix: (batch, seq_len, seq_len), adjacency matrix
        """
        # input: (batch, seq_len, arc_space)
        value_layer = self.value(hidden_states)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(graph_matrix)
        # (batch, seq_len, seq_len) * (batch, seq_len, arc_space) 
        # => (batch, seq_len, arc_space)
        context_layer = torch.matmul(attention_probs.float(), value_layer)
        return context_layer


class GraphAttentionIntermediate(nn.Module):
    def __init__(self, config):
        super(GraphAttentionIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size+config.arc_space, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        super(GraphAttentionLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.graph_attention = GraphAttention(config)
        self.intermediate = GraphAttentionIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, graph_matrix, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        graph_attention_output = self.graph_attention(hidden_states, graph_matrix)
        # (batch, seq_len, hidden_size+arc_space)
        concated_attention_output = torch.cat([attention_output, graph_attention_output],-1)
        intermediate_output = self.intermediate(concated_attention_output)
        layer_output = self.output(intermediate_output, concated_attention_output)
        return layer_output


class GraphAttentionEncoder(nn.Module):
    def __init__(self, config):
        super(GraphAttentionEncoder, self).__init__()
        self.layer = GraphAttentionLayer(config)
        #self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.max_layers)])

    def forward(self, hidden_states, graph_matrices, attention_mask):
        all_encoder_layers = []
        for graph_matrix in graph_matrices:
            hidden_states = self.layer(hidden_states, graph_matrix, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class GraphAttentionModel(nn.Module):
    def __init__(self, config: GraphAttentionConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(GraphAttentionModel, self).__init__()
        self.embeddings = GraphAttentionEmbeddings(config)
        self.encoder = GraphAttentionEncoder(config)

    def forward(self, input_tensor, graph_matrices, attention_mask=None):
        """
        Input:
            input_tensor: (batch, seq_len, input_size)
            graph_matrices: (batch, max_layers, seq_len, seq_len)
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
        all_encoder_layers = self.encoder(embedding_output, graph_matrices, extended_attention_mask)

        return all_encoder_layers

class SelfAttentionConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                input_size=100,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                embedding_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                use_input_layer=False,
                use_sin_position_embedding=False,
                max_position_embeddings=512,
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
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_input_layer = use_input_layer
        self.use_sin_position_embedding = use_sin_position_embedding
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

class SelfAttentionEncoder(nn.Module):
    def __init__(self, config, n_layers=3):
        super(SelfAttentionEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])    

    def forward(self, hidden_states, attention_mask):
        #all_encoder_layers = []
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask)
            #all_encoder_layers.append(hidden_states)
        return hidden_states


class SelfAttentionModel(nn.Module):
    def __init__(self, config: SelfAttentionConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(SelfAttentionModel, self).__init__()
        self.embeddings = GraphAttentionEmbeddings(config)
        self.encoder = BERTEncoder(config)
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


class GraphAttentionV2Config(object):
    """Configuration class to store the configuration of a `GraphAttentionModel`.
    """
    def __init__(self,
                input_size=100,
                hidden_size=768,
                num_graph_attention_layers=1,
                num_attention_heads=1,
                share_params=False,
                only_value_weight=False,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                graph_attention_probs_dropout_prob=0,
                use_input_layer=False,
                use_sin_position_embedding=False,
                max_position_embeddings=512,
                initializer_range=0.02,
                extra_self_attention_layer=False,
                input_self_attention_layer=False,
                num_input_attention_layers=3,
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
            extra_self_attention_layer: whether to use a BERT self attention layer on top
                of graph attention layer
            use_null_att_pos: Whether to enable null attention position at index-0, so that 
                the model learns to attend to it when there is no head yet 
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_graph_attention_layers = num_graph_attention_layers
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.embedding_dropout_prob = hidden_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.graph_attention_probs_dropout_prob = graph_attention_probs_dropout_prob
        self.use_input_layer = use_input_layer
        self.use_sin_position_embedding = use_sin_position_embedding
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.share_params = share_params
        self.only_value_weight = only_value_weight
        self.extra_self_attention_layer = extra_self_attention_layer
        self.input_self_attention_layer = input_self_attention_layer
        self.num_input_attention_layers = num_input_attention_layers
        self.rel_dim = rel_dim
        self.do_encode_rel = do_encode_rel
        self.use_null_att_pos = use_null_att_pos

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


class GraphAttentionV2(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads):
        super(GraphAttentionV2, self).__init__()
        self.only_value_weight = config.only_value_weight
        self.do_encode_rel = config.do_encode_rel
        self.use_null_att_pos = config.use_null_att_pos
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.graph_attention_probs_dropout_prob)
        if not self.only_value_weight:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
        if self.do_encode_rel:
            self.rel_to_hidden = nn.Linear(config.rel_dim, self.attention_head_size)

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
        if self.only_value_weight:
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
            if self.use_null_att_pos:
                graph_matrix = graph_matrix + end_mask.unsqueeze(1)
                #print ("aug matrix:\n", graph_matrix)
            # (batch, seq_len, seq_len), mask out non-ajacency relations
            neg_inf_mask = (1-graph_matrix) * -1e9
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


class GraphAttentionSelfOutputV2(nn.Module):
    def __init__(self, config):
        super(GraphAttentionSelfOutputV2, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GraphAttentionIntermediateV2(nn.Module):
    def __init__(self, config):
        super(GraphAttentionIntermediateV2, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GraphAttentionOutputV2(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GraphAttentionLayerV2(nn.Module):
    def __init__(self, config):
        super(GraphAttentionLayerV2, self).__init__()
        self.input_encoder = None
        if config.input_self_attention_layer:
            self.input_encoder = SelfAttentionEncoder(config, n_layers=config.num_input_attention_layers)
        # information flow in bidirection
        self.fw_graph_attention = GraphAttentionV2(config, config.hidden_size//2, config.num_attention_heads)
        self.bw_graph_attention = GraphAttentionV2(config, config.hidden_size//2, config.num_attention_heads)
        self.self_output = GraphAttentionSelfOutputV2(config)
        self.intermediate = GraphAttentionIntermediateV2(config)
        self.output = BERTOutput(config)
        self.self_attention = None
        if config.extra_self_attention_layer:
            self.self_attention = BERTAttention(config)

    def forward(self, hidden_states, graph_matrix, attention_mask, rel_embeddings=None, 
                end_mask=None):
        if self.input_encoder is not None:
            hidden_states = self.input_encoder(hidden_states, attention_mask)
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
        if self.self_attention is not None:
            # (batch, seq_len, hidden_size), extra self attention layer
            attention_output = self.self_attention(attention_output, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class GraphAttentionEncoderV2(nn.Module):
    def __init__(self, config):
        super(GraphAttentionEncoderV2, self).__init__()
        self.share_params = config.share_params
        self.num_graph_attention_layers = config.num_graph_attention_layers
        layer = GraphAttentionLayerV2(config)
        if self.share_params:
            self.layer = layer
        else:
            self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_graph_attention_layers)])

    def forward(self, hidden_states, graph_matrix, attention_mask, rel_embeddings=None,
                end_mask=None):
        all_encoder_layers = []
        if self.share_params:
            for _ in range(self.num_graph_attention_layers):
                hidden_states = self.layer(hidden_states, graph_matrix, attention_mask, 
                                    rel_embeddings=rel_embeddings, end_mask=end_mask)
                all_encoder_layers.append(hidden_states)
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, graph_matrix, attention_mask, 
                                    rel_embeddings=rel_embeddings, end_mask=end_mask)
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class GraphAttentionModelV2(nn.Module):
    def __init__(self, config: GraphAttentionConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(GraphAttentionModelV2, self).__init__()
        self.embeddings = GraphAttentionEmbeddings(config)
        self.encoder = GraphAttentionEncoderV2(config)

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
