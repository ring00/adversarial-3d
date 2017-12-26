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

    batch_size = cfg.batch_size
    uv = renderer.render(batch_size) * \
        np.asarray([width - 1, height - 1], dtype=np.float32)

    texture = np.asarray(texture).astype(np.float32)[..., :3] / 255.0
    model = AdversarialNet(texture)

    with tf.Session() as sess:

        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(sess, os.path.join(cfg.model_dir, cfg.model_name))

        tf.variables_initializer([model.noise]).run()

        summary, loss, indices = sess.run(
            [model.train_summary, model.loss, model.indices],
            feed_dict={model.uv_mapping: uv}
        )

        print('Loss: {}'.format(loss))
        print('Prediction: {}'.format(indices))

        writer = tf.summary.FileWriter(cfg.logdir)
        writer.add_summary(summary)

if __name__ == '__main__':
    main()
