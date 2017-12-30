#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

from renderer import Renderer
from net import AdversarialNet
from config import cfg

def main():
    texture = Image.open(cfg.texture)
    height, width = texture.size

    renderer = Renderer((299, 299))
    renderer.load_obj(cfg.obj)
    renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )

    texture = np.asarray(texture).astype(np.float32)[..., :3] / 255.0
    model = AdversarialNet(texture)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(sess, os.path.join(cfg.model_dir, cfg.model_name))

        writer = tf.summary.FileWriter(cfg.logdir)

        for i in range(cfg.iterations):
            uv = renderer.render(cfg.batch_size) * \
                np.asarray([width - 1, height - 1], dtype=np.float32)
            summary, loss, indices, diff, _ = sess.run(
                [model.train_summary, model.loss, model.indices, model.diff, model.update],
                feed_dict={model.uv_mapping: uv}
            )

            print('Loss: {}'.format(loss))
            print('Diff: {}'.format(diff.sum()))
            print('Prediction:\n{}'.format(indices))

            writer.add_summary(summary, i)

            if i % 10 == 0:
                adv_texture = np.rint(sess.run([model.adv_texture])[0] * 255)
                adv_texture = Image.fromarray(adv_texture.astype(np.uint8))
                adv_texture.save('{}/adv_{}.jpg'.format(cfg.image_dir, i))

if __name__ == '__main__':
    main()
