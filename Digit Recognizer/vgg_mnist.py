import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features[x], [-1, 28, 28, 1])

    with tf.variable_scope("Conv_Net", reuse=tf.AUTO_REUSE, initializer=tf.truncated_normal_initializer(stddev=0.1)):
        # Convolutional layers
        conv_3x3_64_1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        conv_3x3_64_2 = tf.layers.conv2d(inputs=conv_3x3_64_1, filters=64, kernel_size=[3, 3], padding='same',activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling2d(inputs=conv_3x3_64_2, pool_size=[2, 2], strides=2)

        conv_3x3_128_1 = tf.layers.conv2d(inputs=max_pool_1, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        conv_3x3_128_2 = tf.layers.conv2d(inputs=conv_3x3_128_1, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling2d(inputs=conv_3x3_128_2, pool_size=[2, 2], strides=2)

        # Dense layers
        pool2_flat = tf.reshape(max_pool_2, [-1, 7 * 7 * 128])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, training= mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {"classes": tf.argmax(inputs=logits, axis=1),
                       "probabilities": tf.nn.softmax(logits, axis=1, name="softmax_tensor")}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metrics_op = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=tf.argmax(input=logits, axis=1))}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)