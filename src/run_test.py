import das.tensorflow.estimator as te
t_job= te.TensorFlow(entry_point = '/mnt/hecongqing1/cail/src/AttBert.py', \
                       train_gpu_count = 1,
                       hyperparameters = {'task_name':"task3","do_train":False, "train_path":"../input/SCM_5k.json",
                                          "do_eval":False, "do_predict":True,  "test_path":"../input/SCM_5k.json",
                                          "vocab_file":"./chinese_L-12_H-768_A-12/vocab.txt",
                                          "bert_config_file":"./chinese_L-12_H-768_A-12/bert_config.json",
                                          "init_checkpoint":"../tmp/big_data_attbert_epochs_10_lr2e-5/bert_model.ckpt",
                                          "save_checkpoints_steps":10000000,"max_seq_length":512,"train_batch_size":16,
                                          "learning_rate":2e-5,"num_train_epochs":10.0,
                                          "output_dir":"../tmp/big_data_attbert_epochs_10_lr2e-5/"}
                      )



t_job.fit(base_job_name='test')
