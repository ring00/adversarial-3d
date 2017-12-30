import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import cfg
from nets import inception_v3


class AdversarialNet(object):

    def __init__(self, texture, is_training=True):
        """Construct a network for generating 3D adversarial examples
        Args:
            texture: a numpy array with shape [height, width, 3]
            is_training: training or inference
        """
        super(AdversarialNet, self).__init__()

        self.std_texture = tf.constant(texture, name='texture')
        self.adv_texture = tf.get_variable(
            'adv_texture', initializer=self.std_texture
        )

        self.diff = self.adv_texture - self.std_texture

        self.uv_mapping = tf.placeholder(
            tf.float32, [cfg.batch_size, None, None, 2], name='uv_mapping')

        std_textures = self.repeat(self.std_texture, cfg.batch_size)
        adv_textures = self.repeat(self.adv_texture, cfg.batch_size)

        if cfg.print_error:
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
            std_textures = self.transform(std_textures, multiplier, addend)
            adv_textures = self.transform(adv_textures, multiplier, addend)

        std_images = tf.contrib.resampler.resampler(
            std_textures, self.uv_mapping)
        adv_images = tf.contrib.resampler.resampler(
            adv_textures, self.uv_mapping)

        mask = tf.reduce_all(
            tf.not_equal(self.uv_mapping, 0.0), axis=3, keep_dims=True)
        color = tf.random_uniform(
            [cfg.batch_size, 1, 1, 3], cfg.background_min, cfg.background_max)

        std_images = self.set_backgroud(std_images, mask, color)
        adv_images = self.set_backgroud(adv_images, mask, color)

        if cfg.photo_error:
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
            std_images = self.transform(std_images, multiplier, addend)
            adv_images = self.transform(adv_images, multiplier, addend)

            gaussian_noise = tf.truncated_normal(
                tf.shape(std_images),
                stddev=tf.random_uniform([1], maxval=cfg.stddev)
            )

            std_images += gaussian_noise
            adv_images += gaussian_noise

        # TODO: clip or scale to [0.0, 1.0]?
        # std_images = tf.clip_by_value(std_images, 0, 1)
        # adv_images = tf.clip_by_value(adv_images, 0, 1)
        std_images, adv_images = self.normalize(std_images, adv_images)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            scaled_images = 2.0 * adv_images - 1.0
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                scaled_images, num_classes=1001, is_training=False)

        labels = tf.constant(
            cfg.target, dtype=tf.int64, shape=[cfg.batch_size])

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits_v3)
        l2_loss = tf.reduce_sum(
            tf.square(tf.subtract(std_images, adv_images)), axis=[1, 2, 3])

        self.loss = tf.reduce_mean(cross_entropy + cfg.l2_weight * l2_loss)

        self.train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(self.loss, var_list=[self.adv_texture])

        with tf.control_dependencies([self.train_op]):
            self.update = tf.assign(self.adv_texture, tf.clip_by_value(self.adv_texture, 0, 1))

        train_summary = []
        train_summary.append(tf.summary.image('train/std_images', std_images))
        train_summary.append(tf.summary.image('train/adv_images', adv_images))
        train_summary.append(tf.summary.scalar('train/loss', self.loss))
        train_summary.append(tf.summary.histogram(
            'train/predictions', end_points_v3['Predictions']))

        self.values, self.indices = tf.nn.top_k(end_points_v3['Predictions'], 5)

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

    @staticmethod
    def normalize(x, y):
        minimum = tf.minimum(tf.reduce_min(x, axis=[1, 2], keep_dims=True), tf.reduce_min(y, axis=[1, 2, 3], keep_dims=True))
        maximum = tf.maximum(tf.reduce_max(x, axis=[1, 2], keep_dims=True), tf.reduce_max(y, axis=[1, 2, 3], keep_dims=True))

        minimum = tf.minimum(minimum, 0)
        maximum = tf.maximum(maximum, 1)

        return (x - minimum) / (maximum - minimum), (y - minimum) / (maximum - minimum)
