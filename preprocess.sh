#!/usr/bin/env bash
onmt_preprocess -train_src data/train_clean_file.txt -train_tgt data/train_error_file.txt -valid_src data/valid_clean_file.txt -valid_tgt data/valid_error_file.txt -save_data data/