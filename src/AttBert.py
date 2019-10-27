# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import json
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters


flags.DEFINE_string(
    "bert_config_file", "./chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "task3", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./modelweight/",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "./modelweight/",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_string("train_path", "../input/input.txt", "The train path")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string("eval_path", None, "The dev path")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_string("test_path", "/input/input.txt", "The test path")

flags.DEFINE_integer("train_batch_size", 6, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 6, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 6, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10000000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    pass


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class Task3Processor():
    def get_train_examples(self):
        return self._create_example(FLAGS.train_path, "train")

    def get_dev_examples(self):
        return self._create_example(FLAGS.eval_path, "dev")

    def get_test_examples(self):
        return self._create_example(FLAGS.test_path, "test")

    def get_labels(self):
        return ["B", "C"]

    def _create_example(self, data_path, set_type):
        examples = []
        with open(data_path, encoding='utf-8') as reader:
            for i, line in enumerate(reader):
                lines = json.loads(line)
                for key in ['B', 'C']:
                    guid = "%s-%s" % (set_type, i)
                    text_a = tokenization.convert_to_unicode(lines["A"])
                    text_b = tokenization.convert_to_unicode(lines[key])
                    label = tokenization.convert_to_unicode(key)
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def test_dataframe(self, data_path):
        examples = []
        with open(data_path, encoding='utf-8') as reader:
            for i, line in enumerate(reader):
                lines = json.loads(line)
                for key in ['B', 'C']:
                    guid = i
                    text_a = tokenization.convert_to_unicode(lines["A"])
                    text_b = tokenization.convert_to_unicode(lines[key])
                    label = tokenization.convert_to_unicode(key)
                    examples.append([guid, text_a, text_b, label])
        examples = pd.DataFrame(examples, columns=['guid', 'text_a', 'text_b', "label"])
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    tokens_a = tokens_a[-(int(max_length / 2)):]
    tokens_b = tokens_b[-(max_length - int(max_length / 2)):]
    return tokens_a, tokens_b


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        tokens_a, tokens_b = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # modfix
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)  # modfix
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)  # modfix
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def attention(H):
    """
    利用Attention机制得到句子的向量表示
        hiddenSize:
            最后一层神经元数量
        sequenceLength:
            文本的长度
    """
    sequenceLength = H.shape[1].value
    hiddenSize = H.shape[2].value
    
    # init parameters
    W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
    M = tf.tanh(H)
    # newM = [batch_size, time_step, 1]
    vu = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
    vu = tf.reshape(vu, [-1, sequenceLength])
    
    alpha = tf.nn.softmax(vu) # attention weight
    
    pool_r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, sequenceLength, 1]))
    sequeezeR = tf.squeeze(pool_r) #[batch_size, hidden_size]
    
    sentenceRepren = tf.tanh(sequeezeR)
    pool_output = tf.nn.dropout(sentenceRepren, keep_prob=0.9)
    
    return pool_output


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer =  model.get_sequence_output()
    output_layer = attention(output_layer)
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, 768],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #   init_string = ""
        #   if var.name in initialized_variable_names:
        #     init_string = ", *INIT_FROM_CKPT*"
        #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                   init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                auc = tf.metrics.auc(labels=label_ids, predictions=predictions)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)

                return {
                    "eval_auc": auc,
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "task3": Task3Processor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples()

        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples()
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        test_id = processor.test_dataframe(FLAGS.test_path)

        # with tf.gfile.Open(FLAGS.test_path, "r") as f:
        #   reader = csv.reader(f, delimiter="\t")
        # lines=[]
        # for i,line in enumerate(reader):
        #     if i==0:
        #         continue
        #     else:
        #         lines.append(line[0])
        #
        #
        predicion = []
        for lines in result:
            probabilities = lines["probabilities"][0]
            predicion.append(probabilities)
        test_id['pred'] = predicion
        pred_label = test_id.iloc[test_id.groupby("guid")['pred'].idxmax(), :]['label']

        output_path = "../output/output.txt"
        ouf = open(output_path, "w", encoding="utf8")

        for line in pred_label.tolist():
            print(line, file=ouf)
        ouf.close()

        # output_predict_file = os.path.join(FLAGS.output_dir, "test_results.csv")
        # with tf.gfile.GFile(output_predict_file, "w") as writer:
        #   num_written_lines = 0
        #   tf.logging.info("***** Predict results *****")
        #   for prediction,line in zip(result,lines):
        #     probabilities = prediction["probabilities"]
        #     # if i[0] >= num_actual_predict_examples:
        #     #   break
        #     output_line = line + "\t" + "\t".join(
        #         str(class_probability)
        #         for class_probability in probabilities) + "\n"
        #
        #     writer.write(output_line)
        # num_written_lines += 1
        # assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

"""
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
python main.py \
  --task_name=task3 \
  --do_train=true \
  --train_path=../input/SCM_5k.json \
  --do_eval=true \
  --eval_path=../input/SCM_5k.json \
  --do_predict=False \
  --test_path=../data/test.csv \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --save_checkpoints_steps=10000000 \
  --max_seq_length=500 \
  --train_batch_size=6 \
  --eval_batch_size=6 \
  --predict_batch_size=6 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --output_dir=../tmp/big_data_epochs_10_lr2e-5/


export BERT_BASE_DIR=./chinese_L-12_H-768_A-12

python main.py \
  --task_name=task3 \
  --do_train=true \
  --train_path=../input/SCM_5k.json \
  --do_eval=true \
  --eval_path=../input/SCM_5k.json \
  --do_predict=False \
  --test_path=../data/test.csv \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --save_checkpoints_steps=10000000 \
  --max_seq_length=500 \
  --train_batch_size=6 \
  --eval_batch_size=6 \
  --predict_batch_size=6 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --output_dir=../tmp/big_data_maxpoolig_epochs_10_lr2e-5/


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


export BERT_BASE_DIR=./law
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
  --output_dir=../tmp/big_data_avg_epochs_10_lr2e-5_law/
  
  
  
  

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
  --num_train_epochs=15.0 \
  --output_dir=../tmp/big_data_avg_epochs_15_lr2e-5/
  
  
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
  --output_dir=../tmp/big_data_pool_epochs_10_lr2e-5/
  
"""
