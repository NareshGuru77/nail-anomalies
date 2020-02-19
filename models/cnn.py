import tensorflow as tf
from utils import lr_schedule
from functools import partial

conv = partial(tf.layers.conv2d, kernel_size=[3, 3], padding='same',
                   activation=tf.nn.relu)


def conv_block(in_ts, filters):
    net = conv(inputs=in_ts, filters=filters[0], strides=[1, 1])
    net = tf.layers.batch_normalization(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2, 2])
    net = conv(inputs=net, filters=filters[1], strides=[1, 1])
    net = tf.layers.batch_normalization(net)

    return net


def res_block(in_ts, num_res_layers, filter):
    for i in range(num_res_layers):
        h_i = tf.nn.relu(in_ts)
        h_i = conv(inputs=h_i, filters=filter, strides=[1, 1])
        h_i = conv(inputs=h_i, filters=filter, strides=[1, 1])
        in_ts += h_i

    return in_ts


def custom_network(image):
    net = conv_block(image, [16, 32])
    net = res_block(net, 2, 32)
    net = conv_block(net, [32, 64])
    net = res_block(net, 2, 64)
    net = conv_block(net, [64, 128])
    net = res_block(net, 2, 128)
    net = conv_block(net, [128, 256])

    net = tf.reshape(net, [-1, 16 * 16 * 256])
    net = tf.layers.dense(inputs=net, units=256)

    return tf.layers.dense(inputs=net, units=2)


def model_fn(features, labels, mode, params):
    image = features['image']
    label = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        label = labels['label']
    dimensions = image.get_shape().as_list()
    if len(dimensions) == 3:
        image = tf.expand_dims(image, axis=0)
        if mode != tf.estimator.ModeKeys.PREDICT:
            label = tf.expand_dims(label, axis=0)

    logits = custom_network(image)

    predictions = {
        'class': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    one_hot_l = tf.one_hot(label, depth=2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=one_hot_l))

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = params['learning_rate']

        lr = lr_schedule(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=label, predictions=predictions['class'])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
