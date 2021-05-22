#!/bin/bash

python3 -u train_and_predict.py --bert_ckpt_dir uncased_L-12_H-768_A-12 --dstc8_data_dir dstc8 --dialogues_example_dir dialogues_example --schema_embedding_dir schema_embedding --output_dir dstc8_all --dataset_split test --run_mode predict --task_name dstc8_all --eval_ckpt 70000