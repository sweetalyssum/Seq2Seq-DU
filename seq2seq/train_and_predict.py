# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Baseline model for Schema-guided Dialogue State Tracking.

Adapted from
https://github.com/google-research/bert/blob/master/run_classifier.py
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import sys


import collections
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import embedding_ops
from tensorflow.python.util import nest

import schema
from seq2seq import config
from seq2seq import data_utils
from seq2seq import extract_schema_embedding
from seq2seq import pred_utils
from seq2seq.bert import modeling
from seq2seq.bert import optimization
from seq2seq.bert import tokenization


PAD_ID=0
START_ID=1
SEP_ID = 2
END_ID=3

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# BERT based utterance encoder related flags.
flags.DEFINE_string("bert_ckpt_dir", None,
                    "Directory containing pre-trained BERT checkpoint.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "decode_num_layer", 1,
    "The num of decoder layer")

flags.DEFINE_integer(
    "decoder_hidden_dim", 768,
    "The decoder hidden dim")

flags.DEFINE_integer(
    "decoder_attention_dim", 768,
    "The decoder hidden dim")

flags.DEFINE_integer(
    "bert_dim", 768,
    "The encoder hidden dim of bert")

flags.DEFINE_integer(
    "beam_width", 5,
    "beam_width when testing")

flags.DEFINE_float("dropout_rate", 0.1,
                   "Dropout rate for BERT representations.")

# Hyperparameters and optimization related flags.
flags.DEFINE_integer("train_batch_size", 2, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 7, "Total batch size for predict.")

flags.DEFINE_integer("GPU_num", 4, "number of GPU for training.")

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 80.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

# Input and output paths and other flags.
flags.DEFINE_enum("task_name", None, config.DATASET_CONFIG.keys(),
                  "The name of the task to train.")

flags.DEFINE_string(
    "dstc8_data_dir", None,
    "Directory for the downloaded DSTC8 data, which contains the dialogue files"
    " and schema files of all datasets (eg train, dev)")

flags.DEFINE_enum("run_mode", None, ["train", "predict"],
                  "The mode to run the script in.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "schema_embedding_dir", None,
    "Directory where .npy file for embedding of entities (slots, values,"
    " intents) in the dataset_split's schema are stored.")

flags.DEFINE_string(
    "dialogues_example_dir", None,
    "Directory where tf.record of DSTC8 dialogues data are stored.")

flags.DEFINE_enum("dataset_split", None, ["train", "dev", "test"],
                  "Dataset split for training / prediction.")

flags.DEFINE_string(
    "eval_ckpt", "",
    "Comma separated numbers, each being a step number of model checkpoint"
    " which makes predictions.")

flags.DEFINE_bool(
    "overwrite_dial_file", False,
    "Whether to generate a new Tf.record file saving the dialogue examples.")

flags.DEFINE_bool(
    "overwrite_schema_emb_file", False,
    "Whether to generate a new schema_emb file saving the schemas' embeddings.")

flags.DEFINE_bool(
    "log_data_warnings", False,
    "If True, warnings created using data processing are logged.")

task_name = FLAGS.task_name.lower()
if task_name not in config.DATASET_CONFIG:
    raise ValueError("Task not found: %s" % (task_name))
dataset_config = config.DATASET_CONFIG[task_name]

model_output_dir_with_para = os.path.join(
          FLAGS.output_dir, "output_L{}_H{}_A{}".format(
              FLAGS.decode_num_layer, FLAGS.decoder_hidden_dim, FLAGS.decoder_attention_dim))

# Modified from run_classifier.file_based_input_fn_builder
def _file_based_input_fn_builder(dataset_config, input_dial_file,
                                 schema_embedding_file, is_training,
                                 drop_remainder, batch_size):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  max_num_cat_slot = dataset_config.max_num_cat_slot
  max_num_noncat_slot = dataset_config.max_num_noncat_slot
  max_num_total_slot = max_num_cat_slot + max_num_noncat_slot
  max_num_intent = dataset_config.max_num_intent
  max_utt_len = dataset_config.max_seq_length
  max_num_values = dataset_config.max_num_value_per_cat_slot
  max_decode_seq_len = 2 + 3*max_num_cat_slot + 4*max_num_noncat_slot + 1


  name_to_features = {
      "example_id":
          tf.io.FixedLenFeature([], tf.string),
      "is_real_example":
          tf.io.FixedLenFeature([], tf.int64),
      "service_id":
          tf.io.FixedLenFeature([], tf.int64),
      "utt":
          tf.io.FixedLenFeature([max_utt_len], tf.int64),
      "utt_mask":
          tf.io.FixedLenFeature([max_utt_len], tf.int64),
      "utt_seg":
          tf.io.FixedLenFeature([max_utt_len], tf.int64),
      "cat_slot_num":
          tf.io.FixedLenFeature([], tf.int64),
      # "cat_slot_status":
      #     tf.io.FixedLenFeature([max_num_cat_slot], tf.int64),
      "cat_slot_value_num":
          tf.io.FixedLenFeature([max_num_cat_slot], tf.int64),
      # "cat_slot_value":
      #     tf.io.FixedLenFeature([max_num_cat_slot * 3], tf.int64),
      # "num_active_categorical_slot":
      #     tf.io.FixedLenFeature([], tf.int64),
      "noncat_slot_num":
          tf.io.FixedLenFeature([], tf.int64),
      # "num_active_noncategorical_slot":
      #     tf.io.FixedLenFeature([], tf.int64),
      # "noncategorical_slot_values":
      #     tf.io.FixedLenFeature([max_num_noncat_slot * 4], tf.int64),
      # "noncat_slot_status":
      #     tf.io.FixedLenFeature([max_num_noncat_slot], tf.int64),
      # "noncat_slot_value_start":
      #     tf.io.FixedLenFeature([max_num_noncat_slot], tf.int64),
      # "noncat_slot_value_end":
      #     tf.io.FixedLenFeature([max_num_noncat_slot], tf.int64),
      "noncat_alignment_start":
          tf.io.FixedLenFeature([max_utt_len], tf.int64),
      "noncat_alignment_end":
          tf.io.FixedLenFeature([max_utt_len], tf.int64),
      # "req_slot_num":
      #     tf.io.FixedLenFeature([], tf.int64),
      # "req_slot_status":
      #     tf.io.FixedLenFeature([max_num_total_slot], tf.int64),
      "intent_num":
          tf.io.FixedLenFeature([], tf.int64),
      # "active_intent":
      #     tf.io.FixedLenFeature([2], tf.int64),
      # "intent_status":
      #     tf.io.FixedLenFeature([max_num_intent], tf.int64),
      "output":
          tf.io.FixedLenFeature([1+max_decode_seq_len], tf.int64),
      "dec_output_len":
          tf.io.FixedLenFeature([], tf.int64),  # not contain <start>
  }

  with tf.io.gfile.GFile(schema_embedding_file, "rb") as f:
    schema_data = np.load(f, allow_pickle=True)

  # Convert from list of dict to dict of list
  schema_data_dict = collections.defaultdict(list)
  for service in schema_data:
    schema_data_dict["cat_slot_emb"].append(service["cat_slot_emb"])
    schema_data_dict["cat_slot_value_emb"].append(service["cat_slot_value_emb"])
    schema_data_dict["noncat_slot_emb"].append(service["noncat_slot_emb"])
    schema_data_dict["req_slot_emb"].append(service["req_slot_emb"])
    schema_data_dict["intent_emb"].append(service["intent_emb"])

  def _decode_record(record, name_to_features, schema_tensors):
    """Decodes a record to a TensorFlow example."""

    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    # Here we need to insert schema's entity embedding to each example.

    # Shapes for reference: (all have type tf.float32)
    # "cat_slot_emb": [max_num_cat_slot, hidden_dim]
    # "cat_slot_value_emb": [max_num_cat_slot, max_num_value, hidden_dim]
    # "noncat_slot_emb": [max_num_noncat_slot, hidden_dim]
    # "req_slot_emb": [max_num_total_slot, hidden_dim]
    # "intent_emb": [max_num_intent, hidden_dim]

    service_id = example["service_id"]
    for key, value in schema_tensors.items():
      example[key] = value[service_id]
    return example

  def input_fn():
    """The actual input function."""
    # batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_dial_file)
    # Uncomment for debugging
    # d = d.take(12)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    schema_tensors = {}
    for key, array in schema_data_dict.items():
      schema_tensors[key] = tf.convert_to_tensor(np.asarray(array, np.float32))

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda rec: _decode_record(rec, name_to_features, schema_tensors),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    return d

  return input_fn


class PointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
  """Customized AttentionWrapper for PointerNet."""

  def __init__(self, cell, attention_size, schema_aware_dialogue_embedding, dialogue_aware_schema_embedding, initial_cell_state=None, name=None):
    dialogue_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, schema_aware_dialogue_embedding)
    schema_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, dialogue_aware_schema_embedding)
    # Call super __init__
    super(PointerWrapper, self).__init__(cell,
                                         attention_mechanism=[dialogue_attention_mechanism, schema_attention_mechanism],
                                         attention_layer_size=[attention_size, attention_size],
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)

class PointerBeamSearchDecoder(tf.contrib.seq2seq.BeamSearchDecoder):

    def __init__(self, cell, embedding, start_tokens, end_token,
                 initial_state, beam_width, output_layer=None,
                 length_penalty_weight=0.0, reorder_tensor_arrays=True,
                 coverage_penalty_weight=0.0):
        """Initialize the BeamSearchDecoder.
        Args:
          cell: An `RNNCell` instance.
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.
          initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
          beam_width:  Python integer, the number of beams.
          output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
            to storing the result or sampling.
          length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
          reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell
            state will be reordered according to the beam search path. If the
            `TensorArray` can be reordered, the stacked form will be returned.
            Otherwise, the `TensorArray` will be returned as is. Set this flag to
            `False` if the cell state contains `TensorArray`s that are not amenable
            to reordering.
        Raises:
          TypeError: if `cell` is not an instance of `RNNCell`,
            or `output_layer` is not an instance of `tf.layers.Layer`.
          ValueError: If `start_tokens` is not a vector or
            `end_token` is not a scalar.
        """

        rnn_cell_impl.assert_like_rnncell("cell", cell)  # pylint: disable=protected-access
        # if (output_layer is not None and
        #         not isinstance(output_layer, layers_base.Layer)):
        #     raise TypeError(
        #         "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._output_layer = output_layer
        self._reorder_tensor_arrays = reorder_tensor_arrays
        self._coverage_penalty_weight = coverage_penalty_weight

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        self._batch_size = array_ops.size(start_tokens)
        self._beam_width = beam_width
        self._length_penalty_weight = length_penalty_weight
        self._initial_cell_state = nest.map_structure(
            self._maybe_split_batch_beams, initial_state, self._cell.state_size)
        self._start_tokens = array_ops.tile(
            array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width])
        self._start_inputs = self._embedding_fn(self._start_tokens)

        self._finished = array_ops.one_hot(
            array_ops.zeros([self._batch_size], dtype=dtypes.int32),
            depth=self._beam_width,
            on_value=False,
            off_value=True,
            dtype=dtypes.bool)
        
        
    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            size = 0
            # for attention in self._cell._cells[-1]._attention_mechanisms:
            for attention in self._cell._attention_mechanisms:
                size += attention._alignments_size
            return size


class SchemaGuidedDST(object):
  """Baseline model for schema guided dialogue state tracking."""

  def __init__(self, bert_config, use_one_hot_embeddings):
    self._bert_config = bert_config
    self._use_one_hot_embeddings = use_one_hot_embeddings

  def define_model(self, features, is_training):
    """Define the model computation.

    Args:
      features: A dict mapping feature names to corresponding tensors.
      is_training: A boolean which is True when the model is being trained.

    Returns:
      outputs: A dict mapping output names to corresponding tensors.
    """

    # input: dialogue, noncat_slot, cat_slot, intent, value
    # output: <START> intent, <SEP>, cat_slot, value, <SEP>, noncat_slot, start, end, <SEP>....<END> <PAD>...
    
    # Encode the utterances using BERT.
    # self._encoded_tokens shape (batch, max_seq_len, emb_dim)
    self._encoded_utterance, self._encoded_tokens, self.input_embedding = (
        self._encode_utterances(features, is_training))

    # embedding table
    # shape (batch, max_seq_len, emb_dim)
    dialogue_token_embedding = self.input_embedding
    _, max_seq_len, _ = (dialogue_token_embedding.get_shape().as_list())
    # shape (batch, max_num_noncat_slots, emb_dim)
    noncat_slot_embedding = features["noncat_slot_emb"]
    _, max_num_noncat_slots, _ = (noncat_slot_embedding.get_shape().as_list())
    # shape (batch, max_num_cat_slots, emb_dim)
    cat_slot_embedding = features["cat_slot_emb"]
    # shape (batch, max_num_intent, emb_dim)
    intent_embedding = features["intent_emb"]
    _, max_num_intent, _ = (intent_embedding.get_shape().as_list())
    # shape (batch, max_num_cat_slots*max_num_values, emb_dim)
    value_embedding = features["cat_slot_value_emb"]
    _, max_num_cat_slots, max_num_values, embedding_dim = (
        value_embedding.get_shape().as_list())
    batch_size = tf.shape(value_embedding)[0]
    value_embedding = tf.reshape(value_embedding, [batch_size, max_num_cat_slots * max_num_values, embedding_dim])

    self.dialogue_token_len = max_seq_len
    self.schema_len = max_num_intent + max_num_noncat_slots + max_num_cat_slots + max_num_values*max_num_cat_slots

    # shape
    # (batch_size,
    # max_seq_len + max_num_noncat_slots + max_num_cat_slots + max_num_intents + max_num_cat_slots * max_num_values,
    # embedding_dim)
    input = tf.concat([dialogue_token_embedding, noncat_slot_embedding,
                       cat_slot_embedding, intent_embedding, value_embedding], axis=1)

    # Special token embedding
    # <pad>,<start>,<sep>,<end>
    special_token_embedding = tf.get_variable("special_token_embedding", [4, embedding_dim], tf.float32,
                                              tf.contrib.layers.xavier_initializer())

    # Shape: [batch_size, vocab_size, embedding_dim]
    embedding_table = tf.concat([tf.tile(tf.expand_dims(special_token_embedding, axis=0), [batch_size, 1, 1]), input], axis=1)
    _, self.vocab_size, _ = (embedding_table.get_shape().as_list())
    # Unstack embedding_table
    # Shape: batch_size * [vocab_size, embedding_dim]
    # embedding_table_list = tf.unstack(embedding_table, axis=0)
    
    # shape (batch_size, 1+max_decode_seq_len)
    output = features["output"]
    # shape (batch_size) not contain <start>
    dec_output_len = features["dec_output_len"]
    max_decode_seq_len = 2 + 3*max_num_cat_slots + 4*max_num_noncat_slots + 1
    self.dec_output_mask = tf.sequence_mask(dec_output_len, max_decode_seq_len)

    # Unstack outputs
    # Shape: (1+max_decode_seq_len)*[batch_size]
    outputs_list = tf.unstack(output, axis=1)
    # targets
    # Shape: [batch_size,max_decode_seq_len]
    self.targets = tf.stack(outputs_list[1:], axis=1)

    # decoder input ids
    # Shape: [batch_size, max_decode_seq_len,1]
    dec_input_ids = tf.expand_dims(tf.stack(outputs_list[:-1], axis=1), 2)
    # Look up decoder inputs
    decoder_inputs = []
    if is_training:
        self.batch_size_int = FLAGS.train_batch_size
    else:
        self.batch_size_int = FLAGS.predict_batch_size
    for i in range(self.batch_size_int):
        embedding_table_i = tf.squeeze(tf.slice(embedding_table, [i, 0, 0], [1, self.vocab_size, embedding_dim]))
        dec_input_ids_i = tf.reshape(tf.slice(dec_input_ids, [i, 0, 0], [1, max_decode_seq_len, 1]), [max_decode_seq_len, 1])
        decoder_inputs.append(tf.gather_nd(embedding_table_i, dec_input_ids_i))
    # Shape: [batch_size,max_decode_seq_len,embedding_dim]
    decoder_inputs = tf.stack(decoder_inputs, axis=0)

    # Choose LSTM Cell
    cell = tf.contrib.rnn.LSTMCell

    # attention aware encoder outputs
    # shape
    # (batch_size,
    # max_num_noncat_slots + max_num_cat_slots + max_num_intents + max_num_cat_slots * max_num_values,
    # embedding_dim)
    schema_embedding = tf.concat([noncat_slot_embedding, cat_slot_embedding, intent_embedding, value_embedding], axis=1)
    # shape(batch_size, max_seq_len, emb_dim)
    schema_aware_dialogue_attention = self.get_dialogue_attention(self._encoded_tokens, schema_embedding)
    # shape(batch_size, max_seq_len, 2*emb_dim)
    # schema_aware_dialogue_embedding = tf.concat([self._encoded_tokens, schema_aware_dialogue_attention], axis=-1)
    # shape(batch_size,
    # max_num_noncat_slots + max_num_cat_slots + max_num_intents + max_num_cat_slots * max_num_values,
    # emb_dim)
    dialogue_aware_schema_attention = self.get_schema_attention(schema_embedding, self._encoded_tokens)
    # shape(batch_size,
    # max_num_noncat_slots + max_num_cat_slots + max_num_intents + max_num_cat_slots * max_num_values,
    # 2*emb_dim)
    # dialogue_aware_schema_embedding = tf.concat([schema_embedding, dialogue_aware_schema_attention], axis=-1)

    # add <sep>, <end>
    decode_special_token_embedding = tf.slice(special_token_embedding, [2, 0], [2, embedding_dim])
    # decode_special_token_embedding = tf.tile(decode_special_token_embedding, [1, 2])
    decode_special_token_embedding_reshape = tf.tile(tf.expand_dims(decode_special_token_embedding, axis=0),
                                                     [batch_size, 1, 1])
    self.decode_vocab = tf.concat(
        [decode_special_token_embedding_reshape, schema_aware_dialogue_attention, dialogue_aware_schema_attention],
        axis=1)

    # PointerWrapper
    # shape
    # (batch_size,
    # max_seq_len + max_num_noncat_slots + max_num_cat_slots + max_num_intents + max_num_cat_slots * max_num_values,
    # 2*embedding_dim)
    # memory = tf.concat([schema_aware_dialogue_embedding, dialogue_aware_schema_embedding], axis=1)
    # add <sep>, <end>
    # decode_special_token_embedding = tf.slice(special_token_embedding, [2, 0], [2, embedding_dim])
    # decode_special_token_embedding_reshape = tf.tile(tf.expand_dims(decode_special_token_embedding, axis=0), [batch_size, 1, 1])
    # shape
    # (batch_size,
    # 2 + max_seq_len + max_num_noncat_slots + max_num_cat_slots + max_num_intents + max_num_cat_slots * max_num_values,
    # embedding_dim)
    # memory = tf.concat([decode_special_token_embedding_reshape, memory], axis=1)
    if not is_training:
      # Tile encoder_outiuts
      schema_aware_dialogue_attention = tf.contrib.seq2seq.tile_batch(schema_aware_dialogue_attention, FLAGS.beam_width)
      dialogue_aware_schema_attention = tf.contrib.seq2seq.tile_batch(dialogue_aware_schema_attention, FLAGS.beam_width)
    pointer_cell = PointerWrapper(cell(FLAGS.decoder_hidden_dim), FLAGS.decoder_attention_dim,
                                  schema_aware_dialogue_attention, dialogue_aware_schema_attention)
    # Stack decoder cells if needed
    if FLAGS.decode_num_layer > 1:
        dec_cell = tf.contrib.rnn.MultiRNNCell([cell(FLAGS.decoder_hidden_dim) for _ in range(FLAGS.decode_num_layer - 1)] + [pointer_cell])
    else:
        dec_cell = pointer_cell

    final_outputs = {}
    # Different decoding scenario
    self.w_1 = tf.get_variable("W_1", [FLAGS.decoder_hidden_dim * 2, FLAGS.decoder_hidden_dim], tf.float32,
                          tf.contrib.layers.xavier_initializer())
    self.w_2 = tf.get_variable("W_2", [FLAGS.bert_dim, FLAGS.decoder_hidden_dim], tf.float32,
                          tf.contrib.layers.xavier_initializer())
    self.v = tf.get_variable("v", [FLAGS.decoder_hidden_dim, 1], tf.float32,
                        tf.contrib.layers.xavier_initializer())

    if is_training:
      # Get the maximum sequence length in current batch
      self.cur_batch_max_len = tf.reduce_max(dec_output_len)
      # Training Helper
      helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, dec_output_len)
      # Basic Decoder
      decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, dec_cell.zero_state(batch_size, tf.float32))
      # Decode
      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)
      # logits
      # shape(batch, cur_batch_max_len, emb_dim)
      rnn_output = outputs.rnn_output
      # shape(batch, cur_batch_max_len, vocab-2)
      logits = self.get_logits_for_pointer(rnn_output)
      # predicted_ids_with_logits
      self.predicted_ids_with_logits = tf.nn.top_k(logits)
      # Pad logits to the same shape as targets
      # Shape: [batch_size, max_decode_seq_len, vocab_size-2]
      logits = tf.concat([logits, tf.zeros([batch_size, max_decode_seq_len - self.cur_batch_max_len, self.vocab_size-2])], axis=1)
      final_outputs["logits"] = logits
    else:
        # Tile embedding_table
        tile_embedding_table = tf.tile(tf.expand_dims(embedding_table, 1), [1, FLAGS.beam_width, 1, 1])
    
        # Customize embedding_lookup_fn
        def embedding_lookup(ids):
            # Note the output value of the decoder only ranges 0 to max_input_sequence_len
            # while embedding_table contains two more tokens' values
            # To get around this, shift ids
            # Shape: [batch_size,beam_width]
            ids = ids + 2
            # Shape: [batch_size,beam_width,vocab_size]
            one_hot_ids = tf.cast(tf.one_hot(ids, self.vocab_size), dtype=tf.float32)
            # Shape: [batch_size,beam_width,vocab_size,1]
            one_hot_ids = tf.expand_dims(one_hot_ids, -1)
            # Shape: [batch_size,beam_width,features_size]
            next_inputs = tf.reduce_sum(one_hot_ids * tile_embedding_table, axis=2)
            return next_inputs
    
        # Do a little trick so that we can use 'BeamSearchDecoder'
        shifted_START_ID = START_ID - 2
        shifted_END_ID = END_ID - 2
        # Beam Search Decoder
        decoder = PointerBeamSearchDecoder(dec_cell, embedding_lookup, tf.tile([shifted_START_ID], [batch_size]), shifted_END_ID,
                                           dec_cell.zero_state(batch_size * FLAGS.beam_width, tf.float32),
                                           FLAGS.beam_width, output_layer=self.get_logits_for_pointer)
        # Decode
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_decode_seq_len)
        # predicted_ids
        # Shape: [batch_size, max_output_sequence_len,  beam_width]
        predicted_ids = outputs.predicted_ids
        # Transpose predicted_ids
        # Shape: [batch_size, beam_width, output_sequence_len]
        self.predicted_ids = tf.transpose(predicted_ids, [0, 2, 1])
        final_outputs["predicted_ids"] = self.predicted_ids

    # outputs = {}
    # outputs["logit_output_seq"] = self._get_intents(features)
    # outputs["logit_req_slot_status"] = self._get_requested_slots(features)
    # cat_slot_status, cat_slot_value = self._get_categorical_slot_goals(features)
    # outputs["logit_cat_slot_status"] = cat_slot_status
    # outputs["logit_cat_slot_value"] = cat_slot_value
    # noncat_slot_status, noncat_span_start, noncat_span_end = (
    #     self._get_noncategorical_slot_goals(features))
    # outputs["logit_noncat_slot_status"] = noncat_slot_status
    # outputs["logit_noncat_slot_start"] = noncat_span_start
    # outputs["logit_noncat_slot_end"] = noncat_span_end
    return final_outputs

  def define_loss(self, features, outputs):
    """Obtain the loss of the model."""
    # # Intents.
    # # Shape: (batch_size, max_num_intents + 1).
    # intent_logits = outputs["logit_intent_status"]
    # # Shape: (batch_size, max_num_intents).
    # intent_labels = features["intent_status"]
    # # Add label corresponding to NONE intent.
    # num_active_intents = tf.expand_dims(
    #     tf.reduce_sum(intent_labels, axis=1), axis=1)
    # none_intent_label = tf.ones_like(num_active_intents) - num_active_intents
    # # Shape: (batch_size, max_num_intents + 1).
    # onehot_intent_labels = tf.concat([none_intent_label, intent_labels], axis=1)
    # intent_loss = tf.losses.softmax_cross_entropy(
    #     onehot_intent_labels,
    #     intent_logits,
    #     weights=features["is_real_example"])

    # # Requested slots.
    # # Shape: (batch_size, max_num_slots).
    # requested_slot_logits = outputs["logit_req_slot_status"]
    # requested_slot_labels = features["req_slot_status"]
    # max_num_requested_slots = requested_slot_labels.get_shape().as_list()[-1]
    # weights = tf.sequence_mask(
    #     features["req_slot_num"], maxlen=max_num_requested_slots)
    # # Sigmoid cross entropy is used because more than one slots can be requested
    # # in a single utterance.
    # requested_slot_loss = tf.losses.sigmoid_cross_entropy(
    #     requested_slot_labels, requested_slot_logits, weights=weights)

    # # Categorical slot status.
    # # Shape: (batch_size, max_num_cat_slots, 3).
    # cat_slot_status_logits = outputs["logit_cat_slot_status"]
    # cat_slot_status_labels = features["cat_slot_status"]
    # max_num_cat_slots = cat_slot_status_labels.get_shape().as_list()[-1]
    # one_hot_labels = tf.one_hot(cat_slot_status_labels, 3, dtype=tf.int32)
    # cat_weights = tf.sequence_mask(
    #     features["cat_slot_num"], maxlen=max_num_cat_slots, dtype=tf.float32)
    # cat_slot_status_loss = tf.losses.softmax_cross_entropy(
    #     tf.reshape(one_hot_labels, [-1, 3]),
    #     tf.reshape(cat_slot_status_logits, [-1, 3]),
    #     weights=tf.reshape(cat_weights, [-1]))

    # # Categorical slot values.
    # # Shape: (batch_size, max_num_cat_slots, max_num_slot_values).
    # cat_slot_value_logits = outputs["logit_cat_slot_value"]
    # cat_slot_value_labels = features["cat_slot_value"]
    # max_num_slot_values = cat_slot_value_logits.get_shape().as_list()[-1]
    # one_hot_labels = tf.one_hot(
    #     cat_slot_value_labels, max_num_slot_values, dtype=tf.int32)
    # # Zero out losses for categorical slot value when the slot status is not
    # # active.
    # cat_loss_weight = tf.cast(
    #     tf.equal(cat_slot_status_labels, data_utils.STATUS_ACTIVE), tf.float32)
    # cat_slot_value_loss = tf.losses.softmax_cross_entropy(
    #     tf.reshape(one_hot_labels, [-1, max_num_slot_values]),
    #     tf.reshape(cat_slot_value_logits, [-1, max_num_slot_values]),
    #     weights=tf.reshape(cat_weights * cat_loss_weight, [-1]))

    # # Non-categorical slot status.
    # # Shape: (batch_size, max_num_noncat_slots, 3).
    # noncat_slot_status_logits = outputs["logit_noncat_slot_status"]
    # noncat_slot_status_labels = features["noncat_slot_status"]
    # max_num_noncat_slots = noncat_slot_status_labels.get_shape().as_list()[-1]
    # one_hot_labels = tf.one_hot(noncat_slot_status_labels, 3, dtype=tf.int32)
    # noncat_weights = tf.sequence_mask(
    #     features["noncat_slot_num"],
    #     maxlen=max_num_noncat_slots,
    #     dtype=tf.float32)
    # # Logits for padded (invalid) values are already masked.
    # noncat_slot_status_loss = tf.losses.softmax_cross_entropy(
    #     tf.reshape(one_hot_labels, [-1, 3]),
    #     tf.reshape(noncat_slot_status_logits, [-1, 3]),
    #     weights=tf.reshape(noncat_weights, [-1]))

    # # Non-categorical slot spans.
    # # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
    # span_start_logits = outputs["logit_noncat_slot_start"]
    # span_start_labels = features["noncat_slot_value_start"]
    # max_num_tokens = span_start_logits.get_shape().as_list()[-1]
    # onehot_start_labels = tf.one_hot(
    #     span_start_labels, max_num_tokens, dtype=tf.int32)
    # # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
    # span_end_logits = outputs["logit_noncat_slot_end"]
    # span_end_labels = features["noncat_slot_value_end"]
    # onehot_end_labels = tf.one_hot(
    #     span_end_labels, max_num_tokens, dtype=tf.int32)
    # # Zero out losses for non-categorical slot spans when the slot status is not
    # # active.
    # noncat_loss_weight = tf.cast(
    #     tf.equal(noncat_slot_status_labels, data_utils.STATUS_ACTIVE),
    #     tf.float32)
    # span_start_loss = tf.losses.softmax_cross_entropy(
    #     tf.reshape(onehot_start_labels, [-1, max_num_tokens]),
    #     tf.reshape(span_start_logits, [-1, max_num_tokens]),
    #     weights=tf.reshape(noncat_weights * noncat_loss_weight, [-1]))
    # span_end_loss = tf.losses.softmax_cross_entropy(
    #     tf.reshape(onehot_end_labels, [-1, max_num_tokens]),
    #     tf.reshape(span_end_logits, [-1, max_num_tokens]),
    #     weights=tf.reshape(noncat_weights * noncat_loss_weight, [-1]))

    # Subtract target values by 2
    # because prediction output ranges from 0 to max_input_sequence_len+1
    # while target values are from 0 to max_input_sequence_len + 3
    shifted_targets = (self.targets - 2) * tf.cast(self.dec_output_mask, tf.int32)
    # Shape: [batch_size, max_decode_seq_len, vocab_size-2]
    logits = outputs["logits"]
    # Losses
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=shifted_targets, logits=logits)
    # Total loss
    loss = tf.reduce_sum(losses * tf.cast(self.dec_output_mask, tf.float32)) / FLAGS.train_batch_size

    losses = {
        # "intent_loss": intent_loss,
        # "requested_slot_loss": requested_slot_loss,
        # "cat_slot_status_loss": cat_slot_status_loss,
        # "cat_slot_value_loss": cat_slot_value_loss,
        # "noncat_slot_status_loss": noncat_slot_status_loss,
        # "span_start_loss": span_start_loss,
        # "span_end_loss": span_end_loss,
        "total_loss": loss,
    }
    for loss_name, loss in losses.items():
      tf.summary.scalar(loss_name, loss)
    return sum(losses.values()) / len(losses)

  def define_predictions(self, features, outputs):
    """Define model predictions."""
    predictions = {
        "example_id": features["example_id"],
        "service_id": features["service_id"],
        "is_real_example": features["is_real_example"],
    }
    # Scores are output for each intent.
    # Note that the intent indices are shifted by 1 to account for NONE intent.
    # predictions["intent_status"] = tf.argmax(
    #     outputs["logit_intent_status"], axis=-1)

    # Scores are output for each requested slot.
    # predictions["req_slot_status"] = tf.sigmoid(
    #     outputs["logit_req_slot_status"])

    # For categorical slots, the status of each slot and the predicted value are
    # output.
    # predictions["cat_slot_status"] = tf.argmax(
    #     outputs["logit_cat_slot_status"], axis=-1)
    # predictions["cat_slot_value"] = tf.argmax(
    #     outputs["logit_cat_slot_value"], axis=-1)

    # For non-categorical slots, the status of each slot and the indices for
    # spans are output.
    # predictions["noncat_slot_status"] = tf.argmax(
    #     outputs["logit_noncat_slot_status"], axis=-1)
    # start_scores = tf.nn.softmax(outputs["logit_noncat_slot_start"], axis=-1)
    # end_scores = tf.nn.softmax(outputs["logit_noncat_slot_end"], axis=-1)
    # _, max_num_slots, max_num_tokens = end_scores.get_shape().as_list()
    # batch_size = tf.shape(end_scores)[0]
    # Find the span with the maximum sum of scores for start and end indices.
    # total_scores = (
    #     tf.expand_dims(start_scores, axis=3) +
    #     tf.expand_dims(end_scores, axis=2))
    # Mask out scores where start_index > end_index.
    # start_idx = tf.reshape(tf.range(max_num_tokens), [1, 1, -1, 1])
    # end_idx = tf.reshape(tf.range(max_num_tokens), [1, 1, 1, -1])
    # invalid_index_mask = tf.tile((start_idx > end_idx),
    #                              [batch_size, max_num_slots, 1, 1])
    # total_scores = tf.where(invalid_index_mask, tf.zeros_like(total_scores),
    #                         total_scores)
    # max_span_index = tf.argmax(
    #     tf.reshape(total_scores, [-1, max_num_slots, max_num_tokens**2]),
    #     axis=-1)
    # span_start_index = tf.floordiv(max_span_index, max_num_tokens)
    # span_end_index = tf.floormod(max_span_index, max_num_tokens)
    # predictions["noncat_slot_start"] = span_start_index
    # predictions["noncat_slot_end"] = span_end_index
    # Add inverse alignments.
    predictions["noncat_alignment_start"] = features["noncat_alignment_start"]
    predictions["noncat_alignment_end"] = features["noncat_alignment_end"]

    # Shape: [batch_size, beam_width, output_sequence_len]
    predicted_ids = outputs["predicted_ids"]
    # Shape: [batch_size, output_sequence_len]
    predictions["predicted_seq_ids"] = tf.squeeze(tf.gather(predicted_ids, [0], axis=1))

    return predictions

  def _encode_utterances(self, features, is_training):
    """Encode system and user utterances using BERT."""
    # Optain the embedded representation of system and user utterances in the
    # turn and the corresponding token level representations.
    bert_encoder = modeling.BertModel(
        config=self._bert_config,
        is_training=is_training,
        input_ids=features["utt"],
        input_mask=features["utt_mask"],
        token_type_ids=features["utt_seg"],
        use_one_hot_embeddings=self._use_one_hot_embeddings)
    encoded_utterance = bert_encoder.get_pooled_output()
    encoded_tokens = bert_encoder.get_sequence_output()
    
    # obtain token input embedding
    # shape (batch, seq_len, emb_dim)
    input_embbedding = bert_encoder.word_embedding_output

    # Apply dropout in training mode.
    encoded_utterance = tf.layers.dropout(
        encoded_utterance, rate=FLAGS.dropout_rate, training=is_training)
    encoded_tokens = tf.layers.dropout(
        encoded_tokens, rate=FLAGS.dropout_rate, training=is_training)
    return encoded_utterance, encoded_tokens, input_embbedding

  def get_logits_for_pointer(self, cell_outputs):
      decode_length = tf.shape(cell_outputs)[1]
      cell_outputs_reshape = tf.reshape(tf.tile(tf.matmul(cell_outputs, self.w_1), [1, 1, self.vocab_size-2]),
                 [self.batch_size_int, decode_length*(self.vocab_size-2), FLAGS.decoder_hidden_dim])
      vocab_reshape = tf.tile(tf.matmul(self.decode_vocab, self.w_2), [1, decode_length, 1])
      logits = tf.reshape(tf.matmul(tf.tanh(cell_outputs_reshape + vocab_reshape), self.v), [self.batch_size_int, decode_length, self.vocab_size-2])
      return logits

  def get_dialogue_attention(self, dialogue, schema):
      w_1_d = tf.get_variable("W_1_d", [FLAGS.bert_dim, FLAGS.bert_dim], tf.float32,
                         tf.contrib.layers.xavier_initializer())
      w_2_d = tf.get_variable("W_2_d", [FLAGS.bert_dim, FLAGS.bert_dim], tf.float32,
                            tf.contrib.layers.xavier_initializer())
      dialogue_reshape = tf.reshape(tf.tile(tf.matmul(dialogue, w_1_d), [1, 1, self.schema_len]),
                 [self.batch_size_int, self.dialogue_token_len*self.schema_len, FLAGS.bert_dim])
      schema_reshape = tf.tile(tf.matmul(schema, w_2_d), [1, self.dialogue_token_len, 1])
      v_d = tf.get_variable("v_d", [FLAGS.bert_dim, 1], tf.float32,
                            tf.contrib.layers.xavier_initializer())
      alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(dialogue_reshape + schema_reshape), v_d), [self.batch_size_int, self.dialogue_token_len, self.schema_len]))
      # shape (batch, dialogue_token_len, bert_dim)
      attention = tf.matmul(alpha, schema)
      return attention


  def get_schema_attention(self, schema, dialogue):
      w_1_s = tf.get_variable("W_1_s", [FLAGS.bert_dim, FLAGS.bert_dim], tf.float32,
                         tf.contrib.layers.xavier_initializer())
      w_2_s = tf.get_variable("W_2_s", [FLAGS.bert_dim, FLAGS.bert_dim], tf.float32,
                            tf.contrib.layers.xavier_initializer())
      schema_reshape = tf.reshape(tf.tile(tf.matmul(schema, w_1_s), [1, 1, self.dialogue_token_len]),
                 [self.batch_size_int, self.schema_len*self.dialogue_token_len, FLAGS.bert_dim])
      dialogue_reshape = tf.tile(tf.matmul(dialogue, w_2_s), [1, self.schema_len, 1])
      v_s = tf.get_variable("v_s", [FLAGS.bert_dim, 1], tf.float32,
                            tf.contrib.layers.xavier_initializer())
      alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(dialogue_reshape + schema_reshape), v_s), [self.batch_size_int, self.schema_len, self.dialogue_token_len]))
      # shape (batch, schema_len, bert_dim)
      attention = tf.matmul(alpha, dialogue)
      return attention


  def _get_logits(self, element_embeddings, num_classes, name_scope):
    """Get logits for elements by conditioning on utterance embedding.

    Args:
      element_embeddings: A tensor of shape (batch_size, num_elements,
        embedding_dim).
      num_classes: An int containing the number of classes for which logits are
        to be generated.
      name_scope: The name scope to be used for layers.

    Returns:
      A tensor of shape (batch_size, num_elements, num_classes) containing the
      logits.
    """
    _, num_elements, embedding_dim = element_embeddings.get_shape().as_list()
    # Project the utterance embeddings.
    utterance_proj = tf.keras.layers.Dense(
        units=embedding_dim,
        activation=modeling.gelu,
        name="{}_utterance_proj".format(name_scope))
    utterance_embedding = utterance_proj(self._encoded_utterance)
    # Combine the utterance and element embeddings.
    repeat_utterance_embeddings = tf.tile(
        tf.expand_dims(utterance_embedding, axis=1), [1, num_elements, 1])
    utterance_element_emb = tf.concat(
        [repeat_utterance_embeddings, element_embeddings], axis=2)
    # Project the combined embeddings to obtain logits.
    layer_1 = tf.keras.layers.Dense(
        units=embedding_dim,
        activation=modeling.gelu,
        name="{}_projection_1".format(name_scope))
    layer_2 = tf.keras.layers.Dense(
        units=num_classes, name="{}_projection_2".format(name_scope))
    return layer_2(layer_1(utterance_element_emb))

  def _get_intents(self, features):
    """Obtain logits for intents."""
    intent_embeddings = features["intent_emb"]
    # Add a trainable vector for the NONE intent.
    _, max_num_intents, embedding_dim = intent_embeddings.get_shape().as_list()
    null_intent_embedding = tf.get_variable(
        "null_intent_embedding",
        shape=[1, 1, embedding_dim],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    batch_size = tf.shape(intent_embeddings)[0]
    repeated_null_intent_embedding = tf.tile(null_intent_embedding,
                                             [batch_size, 1, 1])
    intent_embeddings = tf.concat(
        [repeated_null_intent_embedding, intent_embeddings], axis=1)

    logits = self._get_logits(intent_embeddings, 1, "intents")
    # Shape: (batch_size, max_intents + 1)
    logits = tf.squeeze(logits, axis=-1)
    # Mask out logits for padded intents. 1 is added to account for NONE intent.
    mask = tf.sequence_mask(
        features["intent_num"] + 1, maxlen=max_num_intents + 1)
    negative_logits = -0.7 * tf.ones_like(logits) * logits.dtype.max
    return tf.where(mask, logits, negative_logits)

  def _get_requested_slots(self, features):
    """Obtain logits for requested slots."""
    slot_embeddings = features["req_slot_emb"]
    logits = self._get_logits(slot_embeddings, 1, "requested_slots")
    return tf.squeeze(logits, axis=-1)

  def _get_categorical_slot_goals(self, features):
    """Obtain logits for status and values for categorical slots."""
    # Predict the status of all categorical slots.
    slot_embeddings = features["cat_slot_emb"]
    status_logits = self._get_logits(slot_embeddings, 3,
                                     "categorical_slot_status")

    # Predict the goal value.

    # Shape: (batch_size, max_categorical_slots, max_categorical_values,
    # embedding_dim).
    value_embeddings = features["cat_slot_value_emb"]
    _, max_num_slots, max_num_values, embedding_dim = (
        value_embeddings.get_shape().as_list())
    value_embeddings_reshaped = tf.reshape(
        value_embeddings, [-1, max_num_slots * max_num_values, embedding_dim])
    value_logits = self._get_logits(value_embeddings_reshaped, 1,
                                    "categorical_slot_values")
    # Reshape to obtain the logits for all slots.
    value_logits = tf.reshape(value_logits, [-1, max_num_slots, max_num_values])
    # Mask out logits for padded slots and values because they will be
    # softmaxed.
    mask = tf.sequence_mask(
        features["cat_slot_value_num"], maxlen=max_num_values)
    negative_logits = -0.7 * tf.ones_like(value_logits) * value_logits.dtype.max
    value_logits = tf.where(mask, value_logits, negative_logits)
    return status_logits, value_logits

  def _get_noncategorical_slot_goals(self, features):
    """Obtain logits for status and slot spans for non-categorical slots."""
    # Predict the status of all non-categorical slots.
    slot_embeddings = features["noncat_slot_emb"]
    max_num_slots = slot_embeddings.get_shape().as_list()[1]
    status_logits = self._get_logits(slot_embeddings, 3,
                                     "noncategorical_slot_status")

    # Predict the distribution for span indices.
    token_embeddings = self._encoded_tokens
    max_num_tokens = token_embeddings.get_shape().as_list()[1]
    tiled_token_embeddings = tf.tile(
        tf.expand_dims(token_embeddings, 1), [1, max_num_slots, 1, 1])
    tiled_slot_embeddings = tf.tile(
        tf.expand_dims(slot_embeddings, 2), [1, 1, max_num_tokens, 1])
    # Shape: (batch_size, max_num_slots, max_num_tokens, 2 * embedding_dim).
    slot_token_embeddings = tf.concat(
        [tiled_slot_embeddings, tiled_token_embeddings], axis=3)

    # Project the combined embeddings to obtain logits.
    embedding_dim = slot_embeddings.get_shape().as_list()[-1]
    layer_1 = tf.keras.layers.Dense(
        units=embedding_dim,
        activation=modeling.gelu,
        name="noncat_spans_layer_1")
    layer_2 = tf.keras.layers.Dense(units=2, name="noncat_spans_layer_2")
    # Shape: (batch_size, max_num_slots, max_num_tokens, 2)
    span_logits = layer_2(layer_1(slot_token_embeddings))

    # Mask out invalid logits for padded tokens.
    token_mask = features["utt_mask"]  # Shape: (batch_size, max_num_tokens).
    token_mask = tf.cast(token_mask, tf.bool)
    tiled_token_mask = tf.tile(
        tf.expand_dims(tf.expand_dims(token_mask, 1), 3),
        [1, max_num_slots, 1, 2])
    negative_logits = -0.7 * tf.ones_like(span_logits) * span_logits.dtype.max
    span_logits = tf.where(tiled_token_mask, span_logits, negative_logits)
    # Shape of both tensors: (batch_size, max_num_slots, max_num_tokens).
    span_start_logits, span_end_logits = tf.unstack(span_logits, axis=3)
    return status_logits, span_start_logits, span_end_logits


# Modified from run_classifier.model_fn_builder
def _model_fn_builder(bert_config, init_checkpoint, learning_rate,
                      num_train_steps, num_warmup_steps, use_tpu,
                      use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    schema_guided_dst = SchemaGuidedDST(bert_config, use_one_hot_embeddings)
    outputs = schema_guided_dst.define_model(features, is_training)
    if is_training:
      total_loss = schema_guided_dst.define_loss(features, outputs)
    else:
      total_loss = tf.constant(0.0)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)
      global_step = tf.train.get_or_create_global_step()
      logged_tensors = {
          "global_step": global_step,
          "total_loss": total_loss,
      }
      
      # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
      #     mode=mode,
      #     loss=total_loss,
      #     train_op=train_op,
      #     scaffold_fn=scaffold_fn,
      #     training_hooks=[
      #         tf.train.LoggingTensorHook(logged_tensors, every_n_iter=5)
      #     ])

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[
              tf.train.LoggingTensorHook(logged_tensors, every_n_iter=5)
          ])


    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, scaffold_fn=scaffold_fn)

    else:  # mode == tf.estimator.ModeKeys.PREDICT
      predictions = schema_guided_dst.define_predictions(features, outputs)
      # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
      #     mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions)

    return output_spec

  return model_fn


def _create_dialog_examples(processor, dial_file):
  """Create dialog examples and save in the file."""
  if not tf.io.gfile.exists(FLAGS.dialogues_example_dir):
    tf.io.gfile.makedirs(FLAGS.dialogues_example_dir)
  frame_examples = processor.get_dialog_examples(FLAGS.dataset_split)
  data_utils.file_based_convert_examples_to_features(frame_examples,
                                                     processor.dataset_config,
                                                     dial_file)


def _create_schema_embeddings(bert_config, schema_embedding_file,
                              dataset_config):
  """Create schema embeddings and save it into file."""
  if not tf.io.gfile.exists(FLAGS.schema_embedding_dir):
    tf.io.gfile.makedirs(FLAGS.schema_embedding_dir)
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  schema_emb_run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  schema_json_path = os.path.join(FLAGS.dstc8_data_dir, FLAGS.dataset_split,
                                  "schema.json")
  schemas = schema.Schema(schema_json_path)

  # Prepare BERT model for embedding a natural language descriptions.
  bert_init_ckpt = os.path.join(FLAGS.bert_ckpt_dir, "bert_model.ckpt")
  schema_emb_model_fn = extract_schema_embedding.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=bert_init_ckpt,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  schema_emb_estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=schema_emb_model_fn,
      config=schema_emb_run_config,
      predict_batch_size=FLAGS.predict_batch_size)
  vocab_file = os.path.join(FLAGS.bert_ckpt_dir, "vocab.txt")
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)
  emb_generator = extract_schema_embedding.SchemaEmbeddingGenerator(
      tokenizer, schema_emb_estimator, FLAGS.max_seq_length)
  emb_generator.save_embeddings(schemas, schema_embedding_file, dataset_config)


def get_num_params(estimator):
    total_parameters = 0
    for variable in estimator.get_variable_names():
        variable_value = estimator.get_variable_value(variable)
        variable_parameters = variable_value.size
        total_parameters += variable_parameters
    return total_parameters


def main(_):
  vocab_file = os.path.join(FLAGS.bert_ckpt_dir, "vocab.txt")
  task_name = FLAGS.task_name.lower()
  if task_name not in config.DATASET_CONFIG:
    raise ValueError("Task not found: %s" % (task_name))
  dataset_config = config.DATASET_CONFIG[task_name]
  processor = data_utils.Dstc8DataProcessor(
      FLAGS.dstc8_data_dir,
      dataset_config=dataset_config,
      vocab_file=vocab_file,
      do_lower_case=FLAGS.do_lower_case,
      max_seq_length=FLAGS.max_seq_length,
      log_data_warnings=FLAGS.log_data_warnings)

  # Generate the dialogue examples if needed or specified.
  dial_file_name = "{}_{}_examples.tf_record".format(task_name,
                                                     FLAGS.dataset_split)
  dial_file = os.path.join(FLAGS.dialogues_example_dir, dial_file_name)
  if not tf.io.gfile.exists(dial_file) or FLAGS.overwrite_dial_file:
    tf.compat.v1.logging.info("Start generating the dialogue examples.")
    _create_dialog_examples(processor, dial_file)
    tf.compat.v1.logging.info("Finish generating the dialogue examples.")

  # Generate the schema embeddings if needed or specified.
  bert_init_ckpt = os.path.join(FLAGS.bert_ckpt_dir, "bert_model.ckpt")
  tokenization.validate_case_matches_checkpoint(
      do_lower_case=FLAGS.do_lower_case, init_checkpoint=bert_init_ckpt)

  bert_config = modeling.BertConfig.from_json_file(
      os.path.join(FLAGS.bert_ckpt_dir, "bert_config.json"))
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  schema_embedding_file = os.path.join(
      FLAGS.schema_embedding_dir,
      "{}_pretrained_schema_embedding.npy".format(FLAGS.dataset_split))
  if (not tf.io.gfile.exists(schema_embedding_file) or
      FLAGS.overwrite_schema_emb_file):
    tf.compat.v1.logging.info("Start generating the schema embeddings.")
    _create_schema_embeddings(bert_config, schema_embedding_file,
                              dataset_config)
    tf.compat.v1.logging.info("Finish generating the schema embeddings.")
  
  # Create estimator for training or inference.
  tf.io.gfile.makedirs(model_output_dir_with_para)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  run_config = tf.estimator.RunConfig(
      model_dir=model_output_dir_with_para,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=None,
      train_distribute=mirrored_strategy)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  # run_config = tf.contrib.tpu.RunConfig(
  #     cluster=tpu_cluster_resolver,
  #     master=FLAGS.master,
  #     model_dir=model_output_dir_with_para,
  #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  #     keep_checkpoint_max=None,
  #     tpu_config=tf.contrib.tpu.TPUConfig(
  #         # Recommended value is number of global steps for next checkpoint.
  #         iterations_per_loop=FLAGS.save_checkpoints_steps,
  #         num_shards=FLAGS.num_tpu_cores,
  #         per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.run_mode == "train":
      num_train_examples = processor.get_num_dialog_examples(FLAGS.dataset_split)
      num_train_steps = int(num_train_examples / FLAGS.train_batch_size *
                            FLAGS.num_train_epochs / FLAGS.GPU_num)
      num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion / FLAGS.GPU_num)

  bert_init_ckpt = os.path.join(FLAGS.bert_ckpt_dir, "bert_model.ckpt")
  model_fn = _model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=bert_init_ckpt,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  # estimator = tf.contrib.tpu.TPUEstimator(
  #     use_tpu=FLAGS.use_tpu,
  #     model_fn=model_fn,
  #     config=run_config,
  #     train_batch_size=FLAGS.train_batch_size,
  #     eval_batch_size=FLAGS.eval_batch_size,
  #     predict_batch_size=FLAGS.predict_batch_size)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  if FLAGS.run_mode == "train":
    # Train the model.
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num dial examples = %d", num_train_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    # params_num = get_num_params(estimator)
    # tf.compat.v1.logging.info("  Num params = %d", params_num)

    train_input_fn = _file_based_input_fn_builder(
        dataset_config=dataset_config,
        input_dial_file=dial_file,
        schema_embedding_file=schema_embedding_file,
        is_training=True,
        drop_remainder=True,
        batch_size=FLAGS.train_batch_size)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  elif FLAGS.run_mode == "predict":
    # Run inference to obtain model predictions.
    num_actual_predict_examples = processor.get_num_dialog_examples(
        FLAGS.dataset_split)

    tf.compat.v1.logging.info("***** Running prediction *****")
    tf.compat.v1.logging.info("  Num actual examples = %d",
                              num_actual_predict_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    # params_num = get_num_params(estimator)
    # tf.compat.v1.logging.info("  Num params = %d", params_num)

    predict_input_fn = _file_based_input_fn_builder(
        dataset_config=dataset_config,
        input_dial_file=dial_file,
        schema_embedding_file=schema_embedding_file,
        is_training=False,
        drop_remainder=FLAGS.use_tpu,
        batch_size=FLAGS.predict_batch_size)

    input_json_files = [
        os.path.join(FLAGS.dstc8_data_dir, FLAGS.dataset_split,
                     "dialogues_{:03d}.json".format(fid))
        for fid in dataset_config.file_ranges[FLAGS.dataset_split]
    ]
    schema_json_file = os.path.join(FLAGS.dstc8_data_dir, FLAGS.dataset_split,
                                    "schema.json")

    ckpt_nums = [num for num in FLAGS.eval_ckpt.split(",") if num]
    if not ckpt_nums:
      raise ValueError("No checkpoints assigned for prediction.")
    for ckpt_num in ckpt_nums:
      tf.compat.v1.logging.info("***** Predict results for %s set *****",
                                FLAGS.dataset_split)

      predictions = estimator.predict(
          input_fn=predict_input_fn,
          checkpoint_path=os.path.join(model_output_dir_with_para,
                                       "model.ckpt-%s" % ckpt_num))
      
      # Write predictions to file in DSTC8 format.
      dataset_mark = os.path.basename(FLAGS.dstc8_data_dir)
      prediction_dir = os.path.join(
          model_output_dir_with_para, "pred_res_{}_{}_{}_{}".format(
              int(ckpt_num), FLAGS.dataset_split, task_name, dataset_mark))
      if not tf.io.gfile.exists(prediction_dir):
        tf.io.gfile.makedirs(prediction_dir)
      pred_utils.write_predictions_to_file(predictions, input_json_files,
                                           schema_json_file, prediction_dir, dataset_config)


if __name__ == "__main__":
  flags.mark_flag_as_required("dstc8_data_dir")
  flags.mark_flag_as_required("bert_ckpt_dir")
  flags.mark_flag_as_required("dataset_split")
  flags.mark_flag_as_required("schema_embedding_dir")
  flags.mark_flag_as_required("dialogues_example_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("output_dir")
  tf.compat.v1.app.run(main)
