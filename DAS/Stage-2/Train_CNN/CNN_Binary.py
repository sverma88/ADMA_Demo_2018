from __future__ import division
import os
import math
import tensorflow as tf
import numpy as np
import scipy.io


seed = 547
np.random.seed(seed)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108,
                 batch_size=64, y_dim=None, c_dim=3, checkpoint_dir=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess

        if not os.path.exists('results/'):
            os.makedirs('results/')

        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width

        self.y_dim = y_dim

        self.c_dim = 3
        self.build_model()

    def build_model(self):

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.inputs = tf.placeholder(tf.float32, [None,self.input_height, self.input_width, self.c_dim], name='real_images')

        inputs = self.inputs

        #### Writing Loss for CNN

        self.cnn_logits = self.CNN(inputs, reuse=False)
        self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.cnn_logits))

        self.CNN_logits_test = self.CNN_test(inputs)
        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.CNN_logits_test), 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32))

        self.pred_label = tf.argmax(tf.nn.softmax(self.CNN_logits_test), 1)
        self.pred_prob = tf.nn.softmax(self.CNN_logits_test)

    def train(self, config):

        ### Optimizer for CNN
        c_optim = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(self.c_loss)
        tf.global_variables_initializer().run()
        split_perc = 20
        binary_labels = np.array([(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)])

        fixed_label = binary_labels[0,0]


        for iter in range(len(binary_labels)):
            iter_label = binary_labels[iter,1]

            test_data, test_labels = self.load_test(split_prec=split_perc, fixed_digit=fixed_label, iter_digit= iter_label )
            test_batches = int(test_data.shape[0] / 100)

            Accuracy = []
            Pred_Label = []

            for cross_fold in range(3):

                tf.global_variables_initializer().run()
                print("Initializing the CNN")

                data_train, data_label = self.load_train(split_prec=split_perc, fixed_digit=fixed_label, iter_digit=iter_label, fold= cross_fold)

                CNN_batches = int(data_train.shape[0] / 100)

                index_left = (data_train.shape[0] % 100)

                Data_left = data_train[-index_left:]
                Label_left = data_label[-index_left:]

                Validation = []
                termination = 0.0
                criteria = 1.0


                for CNN_epoch in range(config.epoch):

                    for idx in range(CNN_batches):
                        batch_images = data_train[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_labels = data_label[idx * self.batch_size:(idx + 1) * self.batch_size]

                        rand_index = np.random.permutation(batch_images.shape[0])
                        batch_images = batch_images[rand_index]
                        batch_labels = batch_labels[rand_index]

                        _ = self.sess.run([c_optim],
                                          feed_dict={
                                              self.inputs: batch_images,
                                              self.y: batch_labels
                                          })


                    ##### Do it for the left over data

                    if(index_left != 0):
                        _ = self.sess.run([c_optim],
                                          feed_dict={
                                              self.inputs: Data_left,
                                              self.y: Label_left
                                          })

                    ##### Check to terminate CNN Iteration

                    if np.mod(CNN_epoch, 10) == 0:
                        print("epoch", CNN_epoch)

                        Val_Loss = 0.0
                        for idx in range(CNN_batches):
                            batch_images = data_train[idx * self.batch_size:(idx + 1) * self.batch_size]
                            batch_labels = data_label[idx * self.batch_size:(idx + 1) * self.batch_size]

                            Val_Loss += self.accuracy.eval({
                                self.inputs: batch_images,
                                self.y: batch_labels
                            })

                        if (index_left != 0):
                            Val_Loss += self.accuracy.eval(
                                              feed_dict={
                                                  self.inputs: Data_left,
                                                  self.y: Label_left
                                              })

                        val = Val_Loss / data_train.shape[0]
                        print("Val_Accuracy", val)
                        Validation.append(val)

                        criteria = round(abs(Validation[-1] - termination), 5)
                        termination = Validation[-1]
                        print(criteria)

                    if criteria <= 1e-5:
                        break

                Predicted = []

                Test_Loss = 0.0
                for idx in range(test_batches):
                    batch_images = test_data[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels = test_labels[idx * self.batch_size:(idx + 1) * self.batch_size]

                    Test_Loss += self.accuracy.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })

                    Predicted.append(self.pred_label.eval({
                        self.inputs: batch_images
                    }))

                Predicted = np.concatenate(Predicted, axis=0)
                Predicted = np.reshape(Predicted,(-1,1))
                

                print("UDTA", Predicted.shape)

                Pred_Label.append(Predicted)

                acc = Test_Loss /2000
                print("Test_Accuracy", acc)
                Accuracy.append(acc)


            #### Save Labels
            Pred_Label = np.concatenate(Pred_Label, axis=1)
            print("Punjab", Pred_Label.shape)
            scipy.io.savemat('results/Accuracy_{}_{}_{}.mat'.format(split_perc, fixed_label, iter_label), mdict={'Accuracy': Accuracy})
            scipy.io.savemat('results/Pred_Labels_{}_{}_{}.mat'.format(split_perc, fixed_label, iter_label), mdict={'Pred_Labels': Pred_Label})

    def CNN(self, image, reuse=False):
        with tf.variable_scope("CNN") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = tf.nn.relu(tf.layers.conv2d(image, filters=50, kernel_size=(3, 3), padding='same', name='CNN_conv1'))

            h1 = tf.nn.relu(tf.layers.conv2d(h0, filters=50, kernel_size=(3, 3), padding='same', name='CNN_conv2'))

            h2 = tf.layers.max_pooling2d(h1, pool_size=(2, 2), strides=(2, 2), padding='valid', name='CNN_pool1')

            h2_drop = tf.layers.dropout(h2, 0.25)

            h3 = tf.nn.relu(
                tf.layers.conv2d(h2_drop, filters=100, kernel_size=(3, 3), padding='same', name='CNN_conv3'))

            h4 = tf.nn.relu(tf.layers.conv2d(h3, filters=100, kernel_size=(3, 3), padding='same', name='CNN_conv4'))

            h5 = tf.layers.max_pooling2d(h4, pool_size=(2, 2), strides=(2, 2), padding='valid', name='CNN_pool2')

            h5_drop = tf.layers.dropout(h5, 0.25)

            h6 = tf.reshape(h5_drop, shape=[-1, 100 * 8 * 8])

            h7 = tf.layers.dense(h6, 512, activation=tf.nn.relu, name='CNN_FC1')

            h8 = tf.layers.dropout(h7, 0.5)

            h9 = tf.layers.dense(h8, 2, name='CNN_FC2')

            return h9

    def CNN_test(self, image):
        with tf.variable_scope("CNN") as scope:
            scope.reuse_variables()

            h0 = tf.nn.relu(tf.layers.conv2d(image, filters=50, kernel_size=(3, 3), padding='same', name='CNN_conv1'))

            h1 = tf.nn.relu(tf.layers.conv2d(h0, filters=50, kernel_size=(3, 3), padding='same', name='CNN_conv2'))

            h2 = tf.layers.max_pooling2d(h1, pool_size=(2, 2), strides=(2, 2), padding='valid', name='CNN_pool1')

            h2_drop = tf.layers.dropout(h2, 1.0)

            h3 = tf.nn.relu(
                tf.layers.conv2d(h2_drop, filters=100, kernel_size=(3, 3), padding='same', name='CNN_conv3'))

            h4 = tf.nn.relu(tf.layers.conv2d(h3, filters=100, kernel_size=(3, 3), padding='same', name='CNN_conv4'))

            h5 = tf.layers.max_pooling2d(h4, pool_size=(2, 2), strides=(2, 2), padding='valid', name='CNN_pool2')

            h5_drop = tf.layers.dropout(h5, 1.0)

            h6 = tf.reshape(h5_drop, shape=[-1, 100 * 8 * 8])

            h7 = tf.layers.dense(h6, 512, activation=tf.nn.relu, name='CNN_FC1')

            h8 = tf.layers.dropout(h7, 1.0)

            h9 = tf.layers.dense(h8, 2, name='CNN_FC2')

            return h9


    def load_test(self,split_prec,fixed_digit,iter_digit):

        mat = scipy.io.loadmat('Data_{}_{}_{}.mat'.format(split_prec,fixed_digit,iter_digit))

        testX = mat['testX']
        testX = np.transpose(np.reshape(testX,[-1,3,32,32]),[0,2,3,1])

        testX = testX/255.

        testY = mat['testY']

        y_vec = np.zeros((testY.shape[0], self.y_dim), dtype=np.float)
        for i, label in enumerate(testY):
            y_vec[i, testY[i]] = 1.0


        return testX, y_vec

    def load_train(self, split_prec, fixed_digit, iter_digit, fold):

        mat = scipy.io.loadmat('Data_{}_{}_{}.mat'.format(split_prec, fixed_digit, iter_digit))

        trainX = mat['True_Images']
        trainX = trainX[0,fold]

        AugX = mat['GAN_Images']
        AugX = AugX[0,fold]

        trainX = np.concatenate((trainX, AugX), axis=0)
        trainX = np.transpose(np.reshape(trainX, [-1, 3, 32, 32]), [0, 2, 3, 1])

        trainX = trainX / 255.

        trainY = mat['True_Labels']
        trainY = trainY[0,fold]

        AugY = mat['GAN_Labels']
        AugY = AugY[0, fold]

        trainY = np.concatenate((trainY, AugY), axis=0)

        y_vec = np.zeros((trainY.shape[0], self.y_dim), dtype=np.float)
        for i, label in enumerate(trainY):
            y_vec[i, trainY[i]] = 1.0


        rand_index = np.random.permutation(trainX.shape[0])

        trainX = trainX[rand_index]
        y_vec = y_vec[rand_index]

        return trainX, y_vec



    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
