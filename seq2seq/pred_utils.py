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

"""Prediction and evaluation-related utility functions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import traceback
import collections
import json
import os

import tensorflow as tf

import schema
from seq2seq import data_utils

PAD_ID=0
START_ID=1
SEP_ID = 2
END_ID=3

REQ_SLOT_THRESHOLD = 0.5


def get_predicted_dialog(dialog, all_predictions, schemas, data_config):
  """Update labels in a dialogue based on model predictions.

  Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_predictions: A dict mapping prediction name to the predicted value. See
      SchemaGuidedDST class for the contents of this dict.
    schemas: A Schema object wrapping all the schemas for the dataset.

  Returns:
    A json object containing the dialogue with labels predicted by the model.
  """
  # Overwrite the labels in the turn with the predictions from the model. For
  # test set, these labels are missing from the data and hence they are added.
  dialog_id = dialog["dialogue_id"]
  # The slot values tracked for each service.
  all_slot_values = collections.defaultdict(dict)
  history_utterance = ""
  for turn_idx, turn in enumerate(dialog["turns"]):
    try:
      if turn["speaker"] == "USER":
        user_utterance = turn["utterance"]
        # system_utterance = (
        #     dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else "")
        turn_id = "{:02d}".format(turn_idx)
        for frame in turn["frames"]:
          predictions = all_predictions[(dialog_id, turn_id, frame["service"])]
          slot_values = all_slot_values[frame["service"]]
          service_schema = schemas.get_service_schema(frame["service"])
          # Remove the slot spans and state if present.
          frame.pop("slots", None)
          frame.pop("state", None)

          # The baseline model doesn't predict slot spans. Only state predictions
          # are added.
          state = {}

          max_seq_length = data_config.max_seq_length
          max_num_cat_slot = data_config.max_num_cat_slot
          max_num_noncat_slot = data_config.max_num_noncat_slot
          max_num_value_per_cat_slot = data_config.max_num_value_per_cat_slot
          max_num_intent = data_config.max_num_intent

          # max_output_sequence_len = 2 + max_num_cat_slot*3 + max_num_noncat_slot*4 + 1

          # input: <pad>, <start>, || <sep>, <end>, dialogue, noncat_slot, cat_slot, intent, value

          # shape (batch_size, max_output_sequence_len)
          # intent, <sep>, cat_slot, value, <sep>, noncat_slot, start, end, <sep>,...<end>, <pad>.....
          predicted_seq_ids = predictions["predicted_seq_ids"]
          predicted_seq_ids = predicted_seq_ids.tolist()
          predicted_seq_ids[-1] = END_ID-2
          cur_max_output_sequence_len = len(predicted_seq_ids)
          decode_len = predicted_seq_ids.index(END_ID-2) + 1
          if decode_len != cur_max_output_sequence_len:
            predicted_seq_ids[-(cur_max_output_sequence_len-decode_len):] = [PAD_ID-2] * (cur_max_output_sequence_len-decode_len)

          # Add prediction for active intent. Offset is subtracted to account for
          # NONE intent.
          intent_id_bias = 2 + max_seq_length + max_num_noncat_slot + max_num_cat_slot
          active_intent_id = predicted_seq_ids[0] - intent_id_bias
          if active_intent_id < len(service_schema.intents) and active_intent_id >= 0 and \
            cur_max_output_sequence_len > 1 and predicted_seq_ids[1] == SEP_ID - 2:
            state["active_intent"] = service_schema.get_intent_from_id(active_intent_id)
          else:
            state["active_intent"] = ""

          # Add prediction for requested slots.
          requested_slots = []
          # for slot_idx, slot in enumerate(service_schema.slots):
          #   if predictions["req_slot_status"][slot_idx] > REQ_SLOT_THRESHOLD:
          #     requested_slots.append(slot)
          state["requested_slots"] = requested_slots

          # Add prediction for user goal (slot values).
          # Categorical slots.
          cat_slot_id_bias = 2 + max_seq_length + max_num_noncat_slot
          value_id_bias = 2 + max_seq_length + max_num_noncat_slot + max_num_cat_slot + max_num_intent
          for slot_idx, slot in enumerate(service_schema.categorical_slots):
            if slot_idx + cat_slot_id_bias in predicted_seq_ids:
              position_index = predicted_seq_ids.index(slot_idx + cat_slot_id_bias)
              if position_index + 2 < cur_max_output_sequence_len and predicted_seq_ids[position_index + 2] == SEP_ID - 2:
                value_idx = predicted_seq_ids[position_index + 1]
                value_idx = value_idx - value_id_bias
                if value_idx < len(service_schema.get_categorical_slot_values(slot)) and value_idx >= 0:
                  slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]
            value_id_bias += max_num_value_per_cat_slot

          # Non-categorical slots.
          noncat_slot_id_bias = 2 + max_seq_length
          dialogue_id_bias = 2
          for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
            if slot_idx + noncat_slot_id_bias in predicted_seq_ids:
              position_index = predicted_seq_ids.index(slot_idx + noncat_slot_id_bias)
              if position_index + 3 < cur_max_output_sequence_len and predicted_seq_ids[position_index + 3] == SEP_ID - 2:
                start_idx = predicted_seq_ids[position_index + 1] - dialogue_id_bias
                end_idx = predicted_seq_ids[position_index + 2] - dialogue_id_bias
                if start_idx == 0 and end_idx == 0:
                  slot_values[slot] = data_utils.STR_DONTCARE
                elif start_idx <= end_idx and start_idx < max_seq_length and end_idx < max_seq_length and start_idx >= 0 and end_idx >= 0:
                  ch_start_idx = predictions["noncat_alignment_start"][start_idx]
                  ch_end_idx = predictions["noncat_alignment_end"][end_idx]
                  if ch_start_idx < 0 and ch_end_idx < 0:
                    # Add span from the history utterance.
                    slot_values[slot] = (
                        history_utterance[-ch_start_idx - 1:-ch_end_idx])
                  elif ch_start_idx > 0 and ch_end_idx > 0:
                    # Add span from the user utterance.
                    slot_values[slot] = (user_utterance[ch_start_idx - 1:ch_end_idx])
          # Create a new dict to avoid overwriting the state in previous turns
          # because of use of same objects.
          state["slot_values"] = {s: [v] for s, v in slot_values.items()}
          frame["state"] = state
    except Exception as e:
      traceback.print_exc()
      print(turn["utterance"])
    history_utterance += turn["utterance"]
  return dialog


def write_predictions_to_file(predictions, input_json_files, schema_json_file,
                              output_dir, data_config):
  """Write the predicted dialogues as json files.

  Args:
    predictions: An iterator containing model predictions. This is the output of
      the predict method in the estimator.
    input_json_files: A list of json paths containing the dialogues to run
      inference on.
    schema_json_file: Path for the json file containing the schemas.
    output_dir: The directory where output json files will be created.
  """
  tf.compat.v1.logging.info("Writing predictions to %s.", output_dir)
  schemas = schema.Schema(schema_json_file)
  # Index all predictions.
  all_predictions = {}
  for idx, prediction in enumerate(predictions):
    if not prediction["is_real_example"]:
      continue
    tf.compat.v1.logging.log_every_n(
        tf.compat.v1.logging.INFO, "Processed %d examples.", 500, idx)
    _, dialog_id, turn_id, service_name = (
        prediction["example_id"].decode("utf-8").split("-"))
    all_predictions[(dialog_id, turn_id, service_name)] = prediction
    predicted_seq_ids = prediction["predicted_seq_ids"].tolist()
    print(idx)
    print(predicted_seq_ids)


  # Read each input file and write its predictions.
  for input_file_path in input_json_files:
    with tf.io.gfile.GFile(input_file_path) as f:
      dialogs = json.load(f)
      pred_dialogs = []
      for d in dialogs:
        pred_dialogs.append(get_predicted_dialog(d, all_predictions, schemas, data_config))
    input_file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir, input_file_name)
    with tf.io.gfile.GFile(output_file_path, "w") as f:
      json.dump(
          pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
