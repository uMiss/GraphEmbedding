#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class MeanAggregator(Layer):
    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False, include_self=True,
                 seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim
        self.include_self = include_self

    def build(self, input_shapes):
        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(),
                                        name='bias_weight')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        if self.include_self:
            concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        else:
            concat_feat = neigh_feat
        concat_mean = tf.reduce_mean(concat_feat, axis=1, keep_dims=False)

        output = tf.matmul(concat_mean, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation and self.include_self:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output,dim=-1)
        output._uses_learning_phase = True
        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.tanh, l2_reg=0,
                 use_bias=False, include_self=True,
                 seed=1024, **kwargs):
        super(AttentionAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim

    def build(self, input_shapes):
        """ attention weight """
        self.attention_weights = self.add_weight(shape=(1, self.input_dim * 2),
                                                 initializer=glorot_uniform(
                                                     seed=self.seed),
                                                 regularizer=l2(self.l2_reg),
                                                 name="attention_weights")
        self.kernel = self.add_weight(shape=(1, self.input_dim, self.input_dim),
                                                 initializer=glorot_uniform(
                                                     seed=self.seed),
                                                 regularizer=l2(self.l2_reg),
                                                 name="kernel")
        self.kernel1 = self.add_weight(shape=(1, self.input_dim, self.input_dim),
                                                 initializer=glorot_uniform(
                                                     seed=self.seed),
                                                 regularizer=l2(self.l2_reg),
                                                 name="kernel1")

        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(),
                                        name='bias_weight')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)


        """attention"""
        dims = tf.shape(neigh_feat)
        batch_size = dims[0]  # type tensor
        # 1.对中心实体和其邻居的线性变换
        kernel = tf.tile(self.kernel, [batch_size, 1, 1])
        kernel1 = tf.tile(self.kernel1, [batch_size, 1, 1])
        node_feat = tf.matmul(node_feat, kernel)
        neigh_feat = tf.matmul(neigh_feat, kernel1)

        # 2.将node_feat与neigh_feat进行concat，
        tile_node_feat = tf.tile(node_feat, [1, neigh_feat.shape[1], 1])
        concat_node_neigh_feat = tf.concat([neigh_feat, tile_node_feat], axis=2)

        # 3.将concat的结果进行加权，并经过leakyrelu
        attention_weight = tf.tile(self.attention_weights, [batch_size, 1])
        attention_weight = tf.reshape(attention_weight, (batch_size, self.input_dim * 2, 1))
        score = tf.nn.leaky_relu(tf.matmul(concat_node_neigh_feat, attention_weight))

        # 4.softmax得到最后的score
        score = tf.transpose(score, (0, 2, 1))
        # print("score.shape ", score.shape)
        final_score = tf.nn.softmax(score, axis=2)
        # print("final_score.shape ", final_score.shape)
        final_score = tf.transpose(final_score, (0, 2, 1))

        # 5.对所有的neigh_feat进行按score进行加权求和
        attention_feat = tf.matmul(tf.transpose(neigh_feat, (0, 2, 1)), final_score)
        attention_feat = tf.squeeze(attention_feat, axis=2)
        # output = tf.transpose(output, (0, 2, 1))

        output = tf.matmul(attention_feat, self.neigh_weights)

        if self.use_bias:
            output += self.bias
        # if self.activation:
        #     output = self.activation(output)

        # output = tf.nn.l2_normalize(output,dim=-1)
        output._uses_learning_phase = True

        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HighwayLayer(tf.keras.layers.Layer):
    """Highway layer."""

    def compute_output_signature(self, input_signature):
        pass

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.0,
                 activation='softmax',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(HighwayLayer, self).__init__()
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.dropout_rate = dropout_rate

        self.shape = (input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[self.output_dim, self.output_dim],
                                      initializer=self.kernel_initializer,
                                      dtype='float32',
                                      trainable=True)
        self.kernel1 = self.add_weight('kernel1',
                                      shape=[self.input_dim, self.output_dim],
                                      initializer=self.kernel_initializer,
                                      dtype='float32',
                                      trainable=True)

    def call(self, inputs, training=True):
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = tf.matmul(input1, self.kernel1)
        gate = tf.matmul(input1, self.kernel)
        gate = tf.keras.activations.tanh(gate)
        if training and self.dropout_rate > 0.0:
            gate = tf.nn.dropout(gate, self.dropout_rate)
        gate = tf.keras.activations.relu(gate)
        output = tf.add(tf.multiply(input2, 1 - gate), tf.multiply(input1, gate))
        return self.activation(output)


def OurMethod(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
              aggregator_type='mean', dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int64)
    neighbor_input = [Input(shape=(l,), dtype=tf.int64) for l in neighbor_num]
    print("Train in OurMethod")

    h = features
    features_list = []
    print("len(neighbor_num):", len(neighbor_num))
    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_classes
            n_hidden = n_classes
        if i == 0:
            aggregator = MeanAggregator
        else:
            aggregator = AttentionAggregator
        h = aggregator(units=n_classes, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type, include_self=False)(
            [h, node_input, neighbor_input[i]])  #
        features_list.append([h, n_hidden])

    # alinet highway_layer
    highway_layer = HighwayLayer(n_classes, n_classes,
                                 dropout_rate=0.0)
    h = highway_layer([features_list[0][0], features_list[1][0]])
    # 整合高阶邻居信息
    # h = highway_layer([h, features_list[2][0]])
    # h = highway_layer([h, features_list[3][0]])

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model


def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 抽样邻居节点
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]  # nodes里每个节点的邻居
    if sample_num:
        if self_loop:
            sample_num -= 1

        samp_neighs = [
            list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
                _sample(neigh, sample_num, replace=True)) for neigh in neighs]  # 采样邻居
        if self_loop:
            samp_neighs = [
                samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # gcn邻居要加上自己

        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))
