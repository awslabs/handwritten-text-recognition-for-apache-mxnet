# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder and decoder usded in sequence-to-sequence learning."""
__all__ = ['TransformerEncoder', 'TransformerDecoder',
           'get_transformer_encoder_decoder']

from functools import partial
import math
import os
import string
import warnings



from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, MultiHeadAttentionCell
import gluonnlp as nlp
import mxnet as mx
from mxnet.gluon import rnn
from mxnet.gluon.block import Block
from mxnet import cpu, gluon
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.loss import SoftmaxCELoss
import numpy as np


ALPHABET = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']+list(' ' + string.ascii_letters + string.digits + string.punctuation)
ALPHABET_INDEX = {letter: index for index, letter in enumerate(ALPHABET)} # { a: 0, b: 1, etc}
EOS = 3
BOS = 2
PAD = 1
UNK = 0

def decode_char(text):
    output = []
    for val in text:
        if val == EOS:
            break
        elif val == PAD or val == BOS:
            continue
        output.append(ALPHABET[int(val)])
    return "".join(output)

def encode_char(text, src=True, max_len=150):
    encoded = np.ones(max_len, dtype='float32') * PAD
    text = text[:max_len-2]
    i = 0
    if not src:
        encoded[0] = BOS
        i = 1
    for letter in text:
        if letter in ALPHABET_INDEX:
            encoded[i] = ALPHABET_INDEX[letter]
        i += 1
    encoded[i] = EOS
    return encoded, np.array([i+1]).astype('float32')


def _list_bcast_where(F, mask, new_val_l, old_val_l):
    """Broadcast where. Implements out[i] = new_val[i] * mask + old_val[i] * (1 - mask)

    Parameters
    ----------
    F : symbol or ndarray
    mask : Symbol or NDArray
    new_val_l : list of Symbols or list of NDArrays
    old_val_l : list of Symbols or list of NDArrays

    Returns
    -------
    out_l : list of Symbols or list of NDArrays
    """
    return [F.broadcast_mul(new_val, mask) + F.broadcast_mul(old_val, 1 - mask)
            for new_val, old_val in zip(new_val_l, old_val_l)]


def _get_cell_type(cell_type):
    """Get the object type of the cell by parsing the input

    Parameters
    ----------
    cell_type : str or type

    Returns
    -------
    cell_constructor: type
        The constructor of the RNNCell
    """
    if isinstance(cell_type, str):
        if cell_type == 'lstm':
            return rnn.LSTMCell
        elif cell_type == 'gru':
            return rnn.GRUCell
        elif cell_type == 'relu_rnn':
            return partial(rnn.RNNCell, activation='relu')
        elif cell_type == 'tanh_rnn':
            return partial(rnn.RNNCell, activation='tanh')
        else:
            raise NotImplementedError
    else:
        return cell_type


def _get_attention_cell(attention_cell, units=None,
                        scaled=True, num_heads=None,
                        use_bias=False, dropout=0.0):
    """

    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell),\
            'attention_cell must be either string or AttentionCell. Received attention_cell={}'\
                .format(attention_cell)
        return attention_cell


def _nested_sequence_last(data, valid_length):
    """

    Parameters
    ----------
    data : nested container of NDArrays/Symbols
        The input data. Each element will have shape (batch_size, ...)
    valid_length : NDArray or Symbol
        Valid length of the sequences. Shape (batch_size,)
    Returns
    -------
    data_last: nested container of NDArrays/Symbols
        The last valid element in the sequence.
    """
    assert isinstance(data, list)
    if isinstance(data[0], (mx.sym.Symbol, mx.nd.NDArray)):
        F = mx.sym if isinstance(data[0], mx.sym.Symbol) else mx.ndarray
        return F.SequenceLast(F.stack(*data, axis=0),
                              sequence_length=valid_length,
                              use_sequence_length=True)
    elif isinstance(data[0], list):
        ret = []
        for i in range(len(data[0])):
            ret.append(_nested_sequence_last([ele[i] for ele in data], valid_length))
        return ret
    else:
        raise NotImplementedError


class Seq2SeqEncoder(Block):
    r"""Base class of the encoders in sequence to sequence learning models.
    """
    def __call__(self, inputs, valid_length=None, states=None):  #pylint: disable=arguments-differ
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
            The input sequence, Shape (batch_size, length, C_in).
        valid_length : NDArray or None, default None
            The valid length of the input sequence, Shape (batch_size,). This is used when the
            input sequences are padded. If set to None, all elements in the sequence are used.
        states : list of NDArrays or None, default None
            List that contains the initial states of the encoder.

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        return super(Seq2SeqEncoder, self).__call__(inputs, valid_length, states)

    def forward(self, inputs, valid_length=None, states=None):  #pylint: disable=arguments-differ
        raise NotImplementedError


class Seq2SeqDecoder(Block):
    r"""Base class of the decoders in sequence to sequence learning models.

    In the forward function, it generates the one-step-ahead decoding output.

    """
    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        r"""Generates the initial decoder states based on the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list of NDArrays
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
        """
        raise NotImplementedError

    def decode_seq(self, inputs, states, valid_length=None):
        r"""Given the inputs and the context computed by the encoder,
        generate the new states. This is usually used in the training phase where we set the inputs
        to be the target sequence.

        Parameters
        ----------
        inputs : NDArray
            The input embeddings. Shape (batch_size, length, C_in)
        states : list
            The initial states of the decoder.
        valid_length : NDArray or None
            valid length of the inputs. Shape (batch_size,)
        Returns
        -------
        output : NDArray
            The output of the decoder. Shape is (batch_size, length, C_out)
        states: list
            The new states of the decoder
        additional_outputs : list
            Additional outputs of the decoder, e.g, the attention weights
        """
        raise NotImplementedError

    def __call__(self, step_input, states):  #pylint: disable=arguments-differ
        r"""One-step decoding of the input

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size, C_in)
        states : list
            The previous states of the decoder
        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        return super(Seq2SeqDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states):  #pylint: disable=arguments-differ
        raise NotImplementedError
        
def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


class PositionwiseFFN(HybridBlock):
    """Structure of the Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
    use_residual : bool
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    activation : str, default 'relu'
        Activation function
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  activation=activation,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Position-wise encoding of the inputs.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)

        Returns
        -------
        outputs : Symbol or NDArray
            Shape (batch_size, length, C_out)
        """
        outputs = self.ffn_1(inputs)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


class TransformerEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mask=None):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the transformer encoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights =\
            self.attention_cell(inputs, inputs, inputs, mask)
        outputs = self.proj(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs


class TransformerDecoderCell(HybridBlock):
    """Structure of the Transformer Decoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerDecoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell_in = _get_attention_cell(attention_cell,
                                                         units=units,
                                                         num_heads=num_heads,
                                                         scaled=scaled,
                                                         dropout=dropout)
            self.attention_cell_inter = _get_attention_cell(attention_cell,
                                                            units=units,
                                                            num_heads=num_heads,
                                                            scaled=scaled,
                                                            dropout=dropout)
            self.proj_in = nn.Dense(units=units, flatten=False,
                                    use_bias=False,
                                    weight_initializer=weight_initializer,
                                    bias_initializer=bias_initializer,
                                    prefix='proj_in_')
            self.proj_inter = nn.Dense(units=units, flatten=False,
                                       use_bias=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       prefix='proj_inter_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size,
                                       units=units,
                                       use_residual=use_residual,
                                       dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)

            self.layer_norm_in = nn.LayerNorm()
            self.layer_norm_inter = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mem_value, mask=None, mem_mask=None):  #pylint: disable=unused-argument
        #  pylint: disable=arguments-differ
        """Transformer Decoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mem_value : Symbol or NDArrays
            Memory value, i.e. output of the encoder. Shape (batch_size, mem_length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)
        mem_mask : Symbol or NDArray or None
            Mask for mem_value. Shape (batch_size, length, mem_length)

        Returns
        -------
        decoder_cell_outputs: list
            Outputs of the decoder cell. Contains:

            - outputs of the transformer decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer decoder cell
        """
        outputs, attention_in_outputs =\
            self.attention_cell_in(inputs, inputs, inputs, mask)
        outputs = self.proj_in(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_in(outputs)
        inputs = outputs
        outputs, attention_inter_outputs = \
            self.attention_cell_inter(inputs, mem_value, mem_value, mem_mask)
        outputs = self.proj_inter(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_inter(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_in_outputs)
            additional_outputs.append(attention_inter_outputs)
        return outputs, additional_outputs


class TransformerEncoder(HybridBlock, Seq2SeqEncoder):
    """Structure of the Transformer Encoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._num_layers = num_layers
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerEncoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        use_residual=use_residual,
                        scaled=scaled,
                        output_attention=output_attention,
                        prefix='transformer%d_' % i))

    def __call__(self, inputs, states=None, valid_length=None): #pylint: disable=arguments-differ
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states and masks
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the transformer encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder
        """
        return super(TransformerEncoder, self).__call__(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None, steps=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray, Shape(batch_size, length, C_in)
        states : list of NDArray
        valid_length : NDArray
        steps : NDArray
            Stores value [0, 1, ..., length].
            It is used for lookup in positional encoding matrix

        Returns
        -------
        outputs : NDArray
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        length = inputs.shape[1]
        if valid_length is not None:
            mask = mx.nd.broadcast_lesser(
                mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=1), axis=1, size=length)
            if states is None:
                states = [mask]
            else:
                states.append(mask)
        inputs = inputs * math.sqrt(inputs.shape[-1])
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        if valid_length is not None:
            step_output, additional_outputs =\
                super(TransformerEncoder, self).forward(inputs, states, valid_length)
        else:
            step_output, additional_outputs =\
                super(TransformerEncoder, self).forward(inputs, states)
        return step_output, additional_outputs

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        states : list of NDArray or Symbol
        valid_length : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        if states is not None:
            steps = states[-1]
            # Positional Encoding
            inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                       self._max_length,
                                                                       self._units), axis=0))
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = inputs
        if valid_length is not None:
            mask = states[-2]
        else:
            mask = None
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mask)
            inputs = outputs
            if self._output_attention:
                additional_outputs.append(attention_weights)
        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)
        return outputs, additional_outputs


class TransformerDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the Transformer Decoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence. This is used for constructing position encoding
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=128, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerDecoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, 'In TransformerDecoder, the units should be divided ' \
                                       'exactly by the number of heads. Received units={}, ' \
                                       'num_heads={}'.format(units, num_heads)
        self._num_layers = num_layers
        self._units = units
        self._hidden_size = hidden_size
        self._num_states = num_heads
        self._max_length = max_length
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerDecoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        scaled=scaled,
                        use_residual=use_residual,
                        output_attention=output_attention,
                        prefix='transformer%d_' % i))

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        """Initialize the state from the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
            The decoder states, includes:

            - mem_value : NDArray
            - mem_masks : NDArray, optional
        """
        mem_value = encoder_outputs
        decoder_states = [mem_value]
        mem_length = mem_value.shape[1]
        if encoder_valid_length is not None:
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(mem_masks)
        self._encoder_valid_length = encoder_valid_length
        return decoder_states

    def decode_seq(self, inputs, states, valid_length=None):
        """Decode the decoder inputs. This function is only used for training.

        Parameters
        ----------
        inputs : NDArray, Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of decoder states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        output : NDArray, Shape (batch_size, length, C_out)
        states : list
            The decoder states, includes:

            - mem_value : NDArray
            - mem_masks : NDArray, optional
        additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        length_array = mx.nd.arange(length, ctx=inputs.context)
        mask = mx.nd.broadcast_lesser_equal(
            length_array.reshape((1, -1)),
            length_array.reshape((-1, 1)))
        if valid_length is not None:
            batch_mask = mx.nd.broadcast_lesser(
                mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_mul(mx.nd.expand_dims(batch_mask, -1),
                                       mx.nd.expand_dims(mask, 0))
        else:
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
        states = [None] + states
        output, states, additional_outputs = self.forward(inputs, states, mask)
        states = states[1:]
        if valid_length is not None:
            output = mx.nd.SequenceMask(output,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return output, states, additional_outputs

    def __call__(self, step_input, states): #pylint: disable=arguments-differ
        """One-step-ahead decoding of the Transformer decoder.

        Parameters
        ----------
        step_input : NDArray
        states : list of NDArray

        Returns
        -------
        step_output : NDArray
            The output of the decoder.
            In the train mode, Shape is (batch_size, length, C_out)
            In the test mode, Shape is (batch_size, C_out)
        new_states: list
            Includes
            - last_embeds : NDArray or None
                It is only given during testing
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        return super(TransformerDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states, mask=None):  #pylint: disable=arguments-differ, missing-docstring
        input_shape = step_input.shape
        mem_mask = None
        # If it is in testing, transform input tensor to a tensor with shape NTC
        # Otherwise remove the None in states.
        if len(input_shape) == 2:
            if self._encoder_valid_length is not None:
                has_last_embeds = len(states) == 3
            else:
                has_last_embeds = len(states) == 2
            if has_last_embeds:
                last_embeds = states[0]
                step_input = mx.nd.concat(last_embeds,
                                          mx.nd.expand_dims(step_input, axis=1),
                                          dim=1)
                states = states[1:]
            else:
                step_input = mx.nd.expand_dims(step_input, axis=1)
        elif states[0] is None:
            states = states[1:]
        has_mem_mask = (len(states) == 2)
        if has_mem_mask:
            _, mem_mask = states
            augmented_mem_mask = mx.nd.expand_dims(mem_mask, axis=1)\
                .broadcast_axes(axis=1, size=step_input.shape[1])
            states[-1] = augmented_mem_mask
        if mask is None:
            length_array = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
            mask = mx.nd.broadcast_lesser_equal(
                length_array.reshape((1, -1)),
                length_array.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0),
                                        axis=0, size=step_input.shape[0])
        steps = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
        states.append(steps)
        step_output, step_additional_outputs = \
            super(TransformerDecoder, self).forward(step_input * math.sqrt(step_input.shape[-1]),  #pylint: disable=too-many-function-args
                                                    states, mask)
        states = states[:-1]
        if has_mem_mask:
            states[-1] = mem_mask
        new_states = [step_input] + states
        # If it is in testing, only output the last one
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states, mask=None, position_weight=None):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol, Shape (batch_size, length, C_in)
        states : list of NDArray or Symbol
        mask : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, length, C_out)
        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)

        """
        has_mem_mask = (len(states) == 3)
        if has_mem_mask:
            mem_value, mem_mask, steps = states
        else:
            mem_value, steps = states
            mem_mask = None
        # Positional Encoding
        step_input = F.broadcast_add(step_input,
                                     F.expand_dims(F.Embedding(steps,
                                                               position_weight,
                                                               self._max_length,
                                                               self._units),
                                                   axis=0))
        step_input = self.dropout_layer(step_input)
        step_input = self.layer_norm(step_input)
        inputs = step_input
        outputs = inputs
        step_additional_outputs = []
        attention_weights_l = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mem_value, mask, mem_mask)
            if self._output_attention:
                attention_weights_l.append(attention_weights)
            inputs = outputs
        if self._output_attention:
            step_additional_outputs.extend(attention_weights_l)
        return outputs, step_additional_outputs


def get_transformer_encoder_decoder(num_layers=2,
                                    num_heads=8, scaled=True,
                                    units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                                    max_src_length=50, max_tgt_length=50,
                                    weight_initializer=None, bias_initializer='zeros',
                                    prefix='transformer_', params=None):
    """Build a pair of Parallel GNMT encoder/decoder

    Parameters
    ----------
    num_layers : int
    num_heads : int
    scaled : bool
    units : int
    hidden_size : int
    dropout : float
    use_residual : bool
    max_src_length : int
    max_tgt_length : int
    weight_initializer : mx.init.Initializer or None
    bias_initializer : mx.init.Initializer or None
    prefix : str, default 'transformer_'
        Prefix for name of `Block`s.
    params : Parameter or None
        Container for weight sharing between layers.
        Created if `None`.

    Returns
    -------
    encoder : TransformerEncoder
    decoder :TransformerDecoder
    """
    encoder = TransformerEncoder(num_layers=num_layers,
                                 num_heads=num_heads,
                                 max_length=max_src_length,
                                 units=units,
                                 hidden_size=hidden_size,
                                 dropout=dropout,
                                 scaled=scaled,
                                 use_residual=use_residual,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix=prefix + 'enc_', params=params)
    decoder = TransformerDecoder(num_layers=num_layers,
                                 num_heads=num_heads,
                                 max_length=max_tgt_length,
                                 units=units,
                                 hidden_size=hidden_size,
                                 dropout=dropout,
                                 scaled=scaled,
                                 use_residual=use_residual,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix=prefix + 'dec_', params=params)
    return encoder, decoder

class Denoiser(gluon.nn.HybridBlock):
    def __init__(self, alphabet_size, embed_size=128, max_src_length=150, max_tgt_length=150, vocab=None, num_heads=8, num_layers=2, shared_embedding=False):
        super(Denoiser, self).__init__()
        encoder, decoder = get_transformer_encoder_decoder(max_src_length=max_src_length, max_tgt_length=max_tgt_length, units=embed_size, num_heads=num_heads, num_layers=num_layers)
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = gluon.nn.HybridSequential()
        output_dim = alphabet_size
        with self.src_embedding.name_scope():
             self.src_embedding.add(gluon.nn.Embedding(input_dim=alphabet_size, output_dim=embed_size),
                                gluon.nn.Dropout(rate=0.5))
        
        
        if shared_embedding and vocab == None:
            self.tgt_embedding = self.src_embedding
        else:
            if vocab != None:
                input_dim, output_dim = vocab.embedding.idx_to_vec.shape
                embedding = gluon.nn.Embedding(input_dim, output_dim)
                embedding.initialize()
                embedding.weight.set_data(vocab.embedding.idx_to_vec)
            else:
                
                embedding = gluon.nn.Embedding(input_dim=alphabet_size, output_dim=embed_size)
                self.tgt_embedding = gluon.nn.HybridSequential()
                with self.tgt_embedding.name_scope():
                    self.tgt_embedding.add(embedding,
                                    gluon.nn.Dropout(rate=0.5))
        
        self.tgt_proj = gluon.nn.Dense(units=output_dim, flatten=False, #params=self.tgt_embedding[0].params,
                                 prefix='tgt_proj_')
        
    def hybrid_forward(self, F, src_seq, tgt_seq, src_valid_length, tgt_valid_length):
        encoder_outputs, encoder_additional_outputs = self.encode(src_seq, valid_length=src_valid_length)
        decoder_states = self.decoder.init_state_from_encoder(encoder_outputs, encoder_valid_length=src_valid_length)
        tgt_embed = self.tgt_embedding(tgt_seq)
        outputs, _, _ = self.decoder.decode_seq(tgt_embed, decoder_states, tgt_valid_length)
        outputs = self.tgt_proj(outputs)
        return outputs
    
    def encode(self, inputs, states=None, valid_length=None):
        return self.encoder(self.src_embedding(inputs), states, valid_length)
    
    def decode_step(self, step_input, states):
        step_output, states, step_additional_outputs = self.decoder(self.tgt_embedding(step_input), states)
        step_output = self.tgt_proj(step_output)
        return step_output, states, step_additional_outputs
    
    def decode_logprob(self, step_input, states):
        out, states, _ = self.decode_step(step_input, states)
        return out, states
    
    
class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length as the input
    """
    def hybrid_forward(self, F, pred, label, valid_length): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size, length, V)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size,)
        """
        if self._sparse_label:
            sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
        else:
            sample_weight = F.ones_like(label)
        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        return super(SoftmaxCEMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)

# pylint: disable=unused-argument
class _SmoothingWithDim(mx.operator.CustomOp):
    def __init__(self, epsilon=0.1, axis=-1):
        super(_SmoothingWithDim, self).__init__(True)
        self._epsilon = epsilon
        self._axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        inputs = in_data[0]
        outputs = ((1 - self._epsilon) * inputs) + (self._epsilon / float(inputs.shape[self._axis]))
        self.assign(out_data[0], req[0], outputs)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], (1 - self._epsilon) * out_grad[0])


@mx.operator.register('_smoothing_with_dim')
class _SmoothingWithDimProp(mx.operator.CustomOpProp):
    def __init__(self, epsilon=0.1, axis=-1):
        super(_SmoothingWithDimProp, self).__init__(True)
        self._epsilon = float(epsilon)
        self._axis = int(axis)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return out_grad

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return _SmoothingWithDim(self._epsilon, self._axis)
# pylint: enable=unused-argument


class LabelSmoothing(HybridBlock):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Parameters
    ----------
    axis : int, default -1
        The axis to smooth.
    epsilon : float, default 0.1
        The epsilon parameter in label smoothing
    sparse_label : bool, default True
        Whether input is an integer array instead of one hot array.
    units : int or None
        Vocabulary size. If units is not given, it will be inferred from the input.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, axis=-1, epsilon=0.1, units=None,
                 sparse_label=True, prefix=None, params=None):
        super(LabelSmoothing, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._epsilon = epsilon
        self._sparse_label = sparse_label
        self._units = units

    def hybrid_forward(self, F, inputs, units=None): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        F
        inputs : Symbol or NDArray
            Shape (batch_size, length) or (batch_size, length, V)
        units : int or None
        Returns
        -------
        smoothed_label : Symbol or NDArray
            Shape (batch_size, length, V)
        """
        if self._sparse_label:
            assert units is not None or self._units is not None, \
                'units needs to be given in function call or ' \
                'instance initialization when sparse_label is False'
            if units is None:
                units = self._units
            inputs = F.one_hot(inputs, depth=units)
        if units is None and self._units is None:
            return F.Custom(inputs, epsilon=self._epsilon, axis=self._axis,
                            op_type='_smoothing_with_dim')
        else:
            if units is None:
                units = self._units
            return ((1 - self._epsilon) * inputs) + (self._epsilon / units)