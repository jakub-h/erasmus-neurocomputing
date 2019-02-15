"""
Created by Jakub Hruska
"""


import tensorflow as tf


def task2():
    input_layer = tf.constant([[1, -1, 0.5]], name="input")
    weights_h_1 = tf.constant([[0.2, -0.1, 0.8], [-0.5, 0.3, 0.7]], name="wh_1")
    weights_h_2 = tf.constant([[0.7, 0.6], [0.9, -0.7]], name="wh_2")
    weights_o = tf.constant([[1.7, -2.7]], name="wo")
    hidden_layer_1 = tf.sigmoid(tf.matmul(weights_h_1, input_layer, transpose_b=True), name="hidden_1")
    hidden_layer_2 = tf.sigmoid(tf.matmul(weights_h_2, hidden_layer_1), name="hidden_2")
    output_layer = tf.sigmoid(tf.matmul(weights_o, hidden_layer_2), name="output")

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("log", sess.graph)

        print(sess.run(output_layer))
        writer.close()


if __name__ == '__main__':
    task2()
