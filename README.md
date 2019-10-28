# CAIL2019
相似案例匹配


# 运行

''' shell
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
CUDA_VISIBLE_DEVICES=0 python main.py \
  --task_name=task3 \
  --do_train=True \
  --train_path=../input/SCM_5k.json \
  --do_eval=False \
  --eval_path=../input/SCM_5k.json \
  --do_predict=False \
  --test_path=../data/test.csv \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --save_checkpoints_steps=10000000 \
  --max_seq_length=512 \
  --train_batch_size=16 \
  --eval_batch_size=6 \
  --predict_batch_size=6 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --output_dir=../tmp/big_data_avg_epochs_10_lr2e-5/
  '''
