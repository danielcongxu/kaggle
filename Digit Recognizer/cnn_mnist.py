import os
import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

VALIDATION_SIZE = 2000
LEARNING_RATE = 1e-3

def cnn_model_fn(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    with tf.variable_scope("Conv_Net", reuse=tf.AUTO_REUSE, initializer=tf.truncated_normal_initializer(stddev=0.1)):
        # Convolutional layers
        conv_3x3_64_1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, bias_initializer=tf.constant_initializer(value=0.1))
        conv_3x3_64_2 = tf.layers.conv2d(inputs=conv_3x3_64_1, filters=64, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, bias_initializer=tf.constant_initializer(value=0.1))
        max_pool_1 = tf.layers.max_pooling2d(inputs=conv_3x3_64_2, pool_size=[2, 2], strides=2)

        conv_5x5_128_1 = tf.layers.conv2d(inputs=max_pool_1, filters=128, kernel_size=[5, 5], padding='same',
                                          activation=tf.nn.relu, bias_initializer=tf.constant_initializer(value=0.1))
        conv_5x5_128_2 = tf.layers.conv2d(inputs=conv_5x5_128_1, filters=128, kernel_size=[5, 5], padding='same',
                                          activation=tf.nn.relu, bias_initializer=tf.constant_initializer(value=0.1))
        max_pool_2 = tf.layers.max_pooling2d(inputs=conv_5x5_128_2, pool_size=[2, 2], strides=2)

        # Dense layers
        pool2_flat = tf.reshape(max_pool_2, [-1, 7 * 7 * 128])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, training= mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {"classes": tf.argmax(input=logits, axis=1),
                       "probabilities": tf.nn.softmax(logits, axis=1, name="softmax_tensor")}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metrics_op = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)

def normalize_data(data):
    return data / np.max(data)

# Load data
train_set = pd.read_csv(r"E:\Kaggle\kaggle\Digit Recognizer\data set\train.csv")
test_set = pd.read_csv(r"E:\Kaggle\kaggle\Digit Recognizer\data set\test.csv")
train_lb = train_set.iloc[:,0].values
train_X = train_set.iloc[:,1:].values
test_X = test_set.iloc[:,:].values
train_X = np.asarray(train_X, np.float32)
test_X = np.asarray(test_X, np.float32)
train_X = normalize_data(train_X)
test_X = normalize_data(test_X)
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X, val_X, train_lb, val_lb = train_X[0:(len(train_X) - VALIDATION_SIZE), :, :, :], train_X[(len(train_X) - VALIDATION_SIZE):, :, :, :], \
                                 train_lb[0:(len(train_lb) - VALIDATION_SIZE)], train_lb[(len(train_lb) - VALIDATION_SIZE):]


mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
tensor_to_log = {"probabilities": "Conv_Net/softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_X},
    y=train_lb,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

# Evaluate the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": val_X},
    y=val_lb,
    num_epochs=1,
    shuffle=False)
eval_resulsts = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(str(eval_resulsts))

# Predict the model
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_X},
    num_epochs=1,
    shuffle=False)
predict_results = mnist_classifier.predict(input_fn=predict_input_fn)

# Output to the submission file
np.savetxt("mnist_submission.csv",
           np.c_[range(1, len(test_X) + 1), [x["classes"] for x in list(predict_results)]],
           delimiter=', ',
           header='ImageId,Label',
           comments='',
           fmt="%d")
