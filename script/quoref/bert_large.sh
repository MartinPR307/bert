#!/usr/bin/env bash
# -*- coding: utf-8 -*- 



# Author: xiaoy li 
# description:
# 


if [[ $1 == "tpu" ]]; then
    REPO_PATH=/home/xiaoyli1110/xiaoya/bert
    export TPU_NAME=tpu1-v3-vm-1
    export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/xiaoya/bert"
    DATA_DIR=gs://xiaoy-data
    SQUAD_DIR=${DATA_DIR}/quoref
    BERT_DIR=${DATA_DIR}/cased_L-24_H-1024_A-16
    OUTPUT_DIR=gs://pretrain_task/bertlarge_quoref
    tpu_zone=europe-west4-a
    gcp_project=xiaoyli-20-04-274510

    python3 ${REPO_PATH}/run_quoref.py \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --do_train=True \
    --train_file=$SQUAD_DIR/quoref-train-v0.1.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/quoref-dev-v0.1.json \
    --train_batch_size=10 \
    --learning_rate=3e-5 \
    --num_train_epochs=10 \
    --max_seq_length=384 \
    --do_lower_case=False \
    --doc_stride=128 \
    --output_dir=${OUTPUT_DIR} \
    --use_tpu=True \
    --tpu_name=$TPU_NAME \
    --tpu_zone=${tpu_zone} \
    --gcp_project=${gcp_project} \
    --version_2_with_negative=True

elif [[ $1 == "gpu" ]]; then 
    REPO_PATH=/home/lixiaoya/bert
    export PYTHONPATH="$PYTHONPATH:/home/lixiaoya/bert"
    DATA_DIR=/xiaoya
    export CUDA_VISIBLE_DEVICES=0,1
    SQUAD_DIR=${DATA_DIR}/reading_comprehension/squad
    BERT_DIR=${DATA_DIR}/pretrain_ckpt/cased_L-12_H-768_A-12
    OUTPUT_DIR=${DATA_DIR}/export_dir/bert_squad2

    mkdir -p ${OUTPUT_DIR}

    python3 ${REPO_PATH}/run_quoref.py \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v2.0.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/dev-v2.0.json \
    --train_batch_size=24 \
    --learning_rate=3e-5 \
    --num_train_epochs=2.0 \
    --max_seq_length=384 \
    --do_lower_case=False \
    --doc_stride=128 \
    --output_dir=${OUTPUT_DIR} \
    --version_2_with_negative=True
else
    echo "You NEED input the deivce signature such as tpu or gpu"
fi




