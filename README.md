# CAIL2019
相似案例匹配


# 任务介绍

本任务是针对多篇法律文书进行相似度的计算和判断。具体来说，对于提供的每份文书事实描述，从两篇候选集文书中找到与询问文书更为相似的一篇文书。
该相似案例匹配的数据只涉及民间借贷这类文书。

# 数据介绍

本任务使用的数据集是来自“中国裁判文书网”公开的法律文本，其中每份数据由三篇法律文本组成。数据总共涉及一万组文书三元对，所有的文书三元组对都一定属于民间借贷案由。对于每篇法律文本，提供该文书的事实描述部分。具体地，文件的每一行对应一组数据，且每行的格式都为一个json数据。

对于每份数据，用(A,B,C)来代表改组数据，其中(A,B,C)均对应某一篇文书。在训练数据中，文书数据A与B的相似度是大于A与C的相似度，即sim(A,B)>sim(A,C)。

# 方案介绍

方案介绍详情见： https://zhuanlan.zhihu.com/p/88207736


# 运行环境

- tensorflow-gpu>=1.10.0


# 模型运行

``` shell
export BERT_BASE_DIR=../chinese_L-12_H-768_A-12
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
```
