import tensorflow as tf

path = "../output/main6593/"
sess = tf.Session()
imported_meta = tf.train.import_meta_graph(path + 'model.ckpt-5000.meta')
imported_meta.restore(sess, path + 'model.ckpt-5000')
my_vars = []
for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, path + 'model.ckpt')
