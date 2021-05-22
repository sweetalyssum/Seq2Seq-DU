#!/bin/bash

python3 -u evaluate.py --dstc8_data_dir dstc8 --prediction_dir dstc8_all/pred_res_70000_test_dstc8_all_dataset --eval_set test --output_metric_file eval.json
