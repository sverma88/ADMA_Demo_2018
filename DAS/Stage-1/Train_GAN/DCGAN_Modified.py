from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cifar10



from ops import *
from utils import *

seed = 547
np.random.seed(seed)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64,output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, learning_rate=0.0002):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        if not os.path.exists('Generated_Data/'):
            os.makedirs('Generated_Data/')


        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        # batch normalizers for the discriminator
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # batch normalizers for the generator
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.Generated_labels = self.gen_labels()

        self.c_dim = 3
        self.Learning_Rate = round(learning_rate, 4)

        self.build_model()

    def build_model(self):

        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')


        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)


        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):

        ### Optimizer for GAN
        d_optim = tf.train.AdamOptimizer(self.Learning_Rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.Learning_Rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        # download the data and extract training and test splits
        cifar10.maybe_download_and_extract()

        Train_Images, Train_Labels = cifar10.load_training_data()
        sio.savemat("./CIFAR-10/Train_Images.mat", mdict={"Train_Images":Train_Images})
        sio.savemat("./CIFAR-10/Train_Labels.mat", mdict={"Train_Labels":Train_Labels})

        Test_Images, Test_Labels = cifar10.load_test_data()
        sio.savemat("./CIFAR-10/Test_Images.mat", mdict={"Test_Images":Test_Images})
        sio.savemat("./CIFAR-10/Test_Labels.mat", mdict={"Test_Labels":Test_Labels})

        print("Executing GAN")


        tf.global_variables_initializer().run()
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        fake_batches = int(self.Generated_labels.shape[0] / self.batch_size)

        # initial parameters for the GAN and the fraction of split
        gan_iter = 1000
        alpha = 0.1

        fake_labels = np.copy(self.Generated_labels)
        fixed_label = 0
        iter_labels = np.arange((fixed_label + 1), 10)


        for fold in range(len(iter_labels)):
            tf.global_variables_initializer().run()
            counter = 0

            Gan_data_train, Gan_data_label = self.load_cifar(seed=seed, fixed_label=fixed_label, iter_label= iter_labels[fold], frac=alpha)

            Gan_batches = int(Gan_data_train.shape[0] / self.batch_size)

            ##### Starting with GAN
            start_time = time.time()

            for gan_epoch in range(gan_iter):

                for idx in xrange(0, Gan_batches):
                    batch_images = Gan_data_train[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels = Gan_data_label[idx * self.batch_size:(idx + 1) * self.batch_size]

                    rand_index = np.random.permutation(batch_images.shape[0])
                    batch_images = batch_images[rand_index]
                    batch_labels = batch_labels[rand_index]
                    batch_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

                    # Update D network
                    _ = self.sess.run([d_optim],
                                      feed_dict={
                                          self.inputs: batch_images/127.5 - 1.,
                                          self.z: batch_z,
                                          self.y: batch_labels
                                      })
                    # Update G network
                    _ = self.sess.run([g_optim],
                                      feed_dict={
                                          self.z: batch_z,
                                          self.y: batch_labels
                                      })


                    counter = counter + 1

                    if np.mod(counter, 100) == 0:

                        errD_fake = self.d_loss_fake.eval({
                            self.z: batch_z,
                            self.y: batch_labels
                        })

                        errD_real = self.d_loss_real.eval({
                            self.inputs: batch_images / 127.5 - 1.,
                            self.y: batch_labels
                        })

                        errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.y: batch_labels
                        })

                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                              % (gan_epoch, idx, Gan_batches, time.time() - start_time, errD_fake + errD_real, errG))

                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: batch_images/127.5 - 1.,
                                self.y: batch_labels
                            })

                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:d}.png'.format(config.sample_dir, gan_epoch, fold))

                ##### GAN Iterations Terminates

            #### Generate fake Images by utilizing GAN
            Generated_Samples = []
            for index in range(fake_batches):
                sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                sample_labels = fake_labels[index * self.batch_size:(index + 1) * self.batch_size]

                samples = self.sess.run(
                    [self.sampler],feed_dict={
                        self.z: sample_z,
                        self.y: sample_labels
                    })


                Generated_Samples.append(samples)

            Generated_Samples = np.reshape(np.concatenate(Generated_Samples, axis=0), (-1, 32, 32, 3))

            ### Save Random Fake Images Generated by GAN to check the quality
            index_images = np.random.randint(10000, size=100)
            random_samples = Generated_Samples[index_images]
            save_images(random_samples, image_manifold_size(random_samples.shape[0]),
                        './{}/fake_{:d}.png'.format(config.sample_dir,fold))

            ##### Inverse Transform the digits to 255
            Generated_Samples = np.round((Generated_Samples + 1.)*127.5).astype(np.float)

            sio.savemat('./Generated_Data/Images_%d_%d_%d.mat' %(alpha, fixed_label, iter_labels[fold]), mdict={'Images': Generated_Samples})
            sio.savemat('./Generated_Data/Labels_%d_%d_%d.mat' %(alpha, fixed_label, iter_labels[fold]), mdict={'Labels': fake_labels})


    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # if not self.y_dim:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            image = conv_cond_concat(image, yb)
            h0 = lrelu(conv2d(image, image.shape[-1],self.df_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim , self.df_dim * 2 , name='d_h1_conv')))
            h1 = conv_cond_concat(h1, yb)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*2 + self.y_dim, self.df_dim * 4, name='d_h2_conv')))
            h2=tf.reshape(h2, [self.batch_size, -1])
            h2= concat([h2, y],1)
            h4 = dense(h2, 4*4*self.df_dim*4 + self.y_dim, 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4


    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            # if not self.y_dim:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # project `z` and reshape
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            z_ = dense(z, 102, self.gf_dim * 4 * 4 *4, 'g_h0_lin')

            h0 = tf.reshape(z_, [-1, 4, 4, self.gf_dim * 4])
            h0 = tf.nn.relu(self.g_bn0(h0))
            h0 = conv_cond_concat(h0, yb)


            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim * 2], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            h1 = conv_cond_concat(h1, yb)


            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim * 1], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
            h2 = conv_cond_concat(h2, yb)

            h4= deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')

            return tf.nn.tanh(h4)


    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            # if not self.y_dim:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # project `z` and reshape
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            z_ = dense(z, 102, self.gf_dim * 4 * 4 *4, 'g_h0_lin')

            h0 = tf.reshape(z_, [-1, 4, 4, self.gf_dim * 4])
            h0 = tf.nn.relu(self.g_bn0(h0))
            h0 = conv_cond_concat(h0, yb)

            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim * 2], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            h1 = conv_cond_concat(h1, yb)


            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim * 1], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
            h2 = conv_cond_concat(h2, yb)


            h4= deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')

            return tf.nn.tanh(h4)


    def load_cifar(self, seed, fixed_label, iter_label,frac):

        mat = sio.loadmat('./CIFAR-10/Train_Images.mat')
        trX = mat['Train_Images'].astype(np.float)
        mat = scipy.io.loadmat('./CIFAR-10/Train_Labels.mat')
        trY = mat['Train_Labels'].astype(np.int32)
        trY = np.reshape(trY, (-1,1))

        np.random.seed(seed)

        rand_index = np.random.permutation(trX.shape[0])

        trX = trX[rand_index]
        trY = trY[rand_index]

        bool_index_1 = np.reshape(np.concatenate([trY == fixed_label], axis=0), trY.shape[0])
        Digits1 = trX[bool_index_1][:]
        total_digits1 = int(len(Digits1)* frac)
        bool_index_2 = np.reshape(np.concatenate([trY == iter_label], axis=0), trY.shape[0])
        Digits2 = trX[bool_index_2][:]
        total_digits2 = int(len(Digits2)* frac)
        Digits1 = Digits1[:total_digits1]
        Digits2 = Digits2[:total_digits2]

        Digits = np.concatenate((Digits1,Digits2), axis=0)

        Binary_One_hot1 = np.zeros((1,2))
        Binary_One_hot1[0,0] = 1
        Binary_One_hot1 = Binary_One_hot1.astype(int)
        Binary_One_hot1 = np.tile(Binary_One_hot1, (total_digits1,1))

        Binary_One_hot2 = np.zeros((1,2))
        Binary_One_hot2[0,1] = 1
        Binary_One_hot2 = Binary_One_hot2.astype(int)
        Binary_One_hot2 = np.tile(Binary_One_hot2, (total_digits2,1))

        Binary_One_hot = np.concatenate((Binary_One_hot1,Binary_One_hot2), axis=0)

        rand_index = np.random.permutation(Digits.shape[0])

        Digits = Digits[rand_index]
        Binary_One_hot = Binary_One_hot[rand_index]

        trainX = Digits
        trainY = Binary_One_hot

        return trainX , trainY



    def gen_labels(self):

        Binary_One_hot = np.zeros((2,2))
        Binary_One_hot[0,0] = 1
        Binary_One_hot[1,1] = 1
        Binary_One_hot = Binary_One_hot.astype(int)

        Binary_One_hot = np.tile(Binary_One_hot, (5000,1))
        fake_vec = Binary_One_hot

        return fake_vec


