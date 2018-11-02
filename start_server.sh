#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

DATA_DIR=./testdata

MODEL_DIR=/home/user_data/zhengzx/mt/model_zh2jp
VOCAB_SOURCE=${MODEL_DIR}/vocab.cn
VOCAB_TARGET=${MODEL_DIR}/vocab.jp
CONFIG_PATHS=$MODEL_DIR/model_configs.yml

BATCH_SIZE=13
BEAM_SIZE=5
DELIMITER=" "
MAXIMUM_LABELS_LENGTH=150
CHAR_LEVEL=false

python -m bin.server \
  --model_dir ${MODEL_DIR} \
  --server_address "127.0.0.1:1234" \
  --config_paths $CONFIG_PATHS \
  --infer "
    batch_size: ${BATCH_SIZE}
    beam_size: ${BEAM_SIZE}
    maximum_labels_length: ${MAXIMUM_LABELS_LENGTH}
    delimiter: ${DELIMITER}
    source_words_vocabulary: ${VOCAB_SOURCE}
    target_words_vocabulary: ${VOCAB_TARGET}
    source_bpecodes:
    target_bpecodes:
    char_level: ${CHAR_LEVEL}"
