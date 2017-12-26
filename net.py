import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import cfg
from nets import inception_v3


class AdversarialNet(object):

    def __init__(self, texture, is_training=True):
        """Construct a network for generating 3D adversarial examples
        Args:
            texture: a numpy array with shape [height, width, 3]
            is_training: traing or inference
        """
        super(AdversarialNet, self).__init__()

        self.in_texture = tf.constant(
            texture, dtype=tf.float32, name='texture')

        self.noise = tf.get_variable(
            'noise', shape=texture.shape, dtype=tf.float32,
            initializer=tf.zeros_initializer)

        self.out_texture = tf.add(self.in_texture, self.noise)

        self.uv_mapping = tf.placeholder(
            tf.float32, [cfg.batch_size, None, None, 2], name='uv_mapping')

        in_textures = self.repeat(self.in_texture, cfg.batch_size)
        out_textures = self.repeat(self.out_texture, cfg.batch_size)

        if cfg.printing_error:
            multiplier = tf.random_uniform(
                [cfg.batch_size, 1, 1, 3],
                cfg.channel_mult_min,
                cfg.channel_mult_max
            )
            addend = tf.random_uniform(
                [cfg.batch_size, 1, 1, 3],
                cfg.channel_add_min,
                cfg.channel_add_max
            )
            in_textures = self.transform(in_textures, multiplier, addend)
            out_textures = self.transform(out_textures, multiplier, addend)

        in_images = tf.contrib.resampler.resampler(
            in_textures, self.uv_mapping)
        out_images = tf.contrib.resampler.resampler(
            out_textures, self.uv_mapping)

        mask = tf.reduce_all(
            tf.not_equal(self.uv_mapping, 0.0), axis=3, keep_dims=True)
        color = tf.random_uniform(
            [cfg.batch_size, 1, 1, 3], cfg.background_min, cfg.background_max)

        in_images = self.set_backgroud(in_images, mask, color)
        out_images = self.set_backgroud(out_images, mask, color)

        if cfg.photography_error:
            multiplier = tf.random_uniform(
                [cfg.batch_size, 1, 1, 1],
                cfg.light_mult_min,
                cfg.light_mult_max
            )
            addend = tf.random_uniform(
                [cfg.batch_size, 1, 1, 1],
                cfg.light_add_min,
                cfg.light_add_max
            )
            in_images = self.transform(in_images, multiplier, addend)
            out_images = self.transform(out_images, multiplier, addend)

            gaussian_noise = tf.random_normal(tf.shape(in_images), mean=0.0, stddev=tf.random_uniform([1], cfg.stddev))

            in_images += gaussian_noise
            out_images += gaussian_noise

        # TODO: clip or scale to [0.0, 1.0]?
        # in_images = tf.clip_by_value(in_images, 0.0, 1.0)
        # out_images = tf.clip_by_value(out_images, 0.0, 1.0)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            scaled = 2.0 * out_images - 1.0
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                scaled, num_classes=1001, is_training=False)

        labels = tf.constant(
            cfg.label, dtype=tf.int64, shape=[cfg.batch_size], name='labels')

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits_v3)
        l2_loss = tf.reduce_sum(
            tf.square(tf.subtract(in_images, out_images)), axis=[1, 2, 3])

        self.loss = tf.reduce_mean(cross_entropy + cfg.l2_weight * l2_loss)

        self.train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(self.loss)

        train_summary = []
        train_summary.append(tf.summary.image('train/in_images', in_images))
        train_summary.append(tf.summary.image('train/out_images', out_images))
        train_summary.append(tf.summary.scalar('train/loss', self.loss))
        train_summary.append(tf.summary.histogram(
            'train/predictions', end_points_v3['Predictions']))

        self.values, self.indices = tf.nn.top_k(
            end_points_v3['Predictions'], 5)

        self.train_summary = tf.summary.merge(train_summary)

    @staticmethod
    def repeat(x, times):
        """Repeat a image multiple times to generate a batch
        Args:
            x: A 3-D tensor with shape [height, width, 3]
        Returns:
            A 4-D tensor with shape [batch_size, height, size, 3]
        """
        return tf.tile(tf.expand_dims(x, 0), [cfg.batch_size, 1, 1, 1])

    @staticmethod
    def transform(x, a, b):
        """Apply transform a * x + b element-wise
        """
        return tf.add(tf.multiply(a, x), b)

    @staticmethod
    def set_backgroud(x, mask, color):
        """Set background color according to a boolean mask
        Args:
            x: A 4-D tensor with shape [batch_size, height, size, 3]
            mask: boolean mask with shape [batch_size, height, width, 1]
            color: background color with shape [batch_size, 1, 1, 3]
        """
        mask = tf.tile(mask, [1, 1, 1, 3])
        inv = tf.logical_not(mask)

        return tf.cast(mask, tf.float32) * x + tf.cast(inv, tf.float32) * color
