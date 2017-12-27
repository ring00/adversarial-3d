import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('l2_weight', 1.0, 'the weighting factor for l2 loss')
flags.DEFINE_integer('label', 1, 'the label for adversarial examples')

############################
#   environment setting    #
############################

flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('model_dir', 'model_dir', 'model directory')
flags.DEFINE_string('model_url', 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz', 'model download link')
flags.DEFINE_string('model_name', 'inception_v3.ckpt', 'name of checkpoint file')

############################
#   renderer setting       #
############################

flags.DEFINE_string('obj', '3d_model/barrel.obj', '.obj file path')
flags.DEFINE_string('texture', '3d_model/barrel.jpg', 'texture file path')

flags.DEFINE_float('camera_distance_min', 2.5, 'minimum camera distance')
flags.DEFINE_float('camera_distance_max', 3.0, 'maximum camera distance')

flags.DEFINE_float('x_translation_min', -0.5, 'minimum translation along x-axis')
flags.DEFINE_float('x_translation_max', 0.5, 'maximum translation along x-axis')

flags.DEFINE_float('y_translation_min', -0.5, 'minimum translation along y-axis')
flags.DEFINE_float('y_translation_max', 0.5, 'maximum translation along y-axis')

############################
# post-processing setting  #
############################

flags.DEFINE_boolean('printing_error', True, 'consider printing error for textures')
flags.DEFINE_boolean('photography_error', False, 'consider photography error for images')

flags.DEFINE_float('background_min', 0.1, 'minimum rgb value for background')
flags.DEFINE_float('background_max', 1.0, 'maximum rgb value for background')

flags.DEFINE_float('light_add_min', -0.15, 'minimum additive lighten/darken')
flags.DEFINE_float('light_add_max', 0.15, 'maximum additive lighten/darken')

flags.DEFINE_float('light_mult_min', 0.5, 'minimum multiplicative lighten/darken')
flags.DEFINE_float('light_mult_max', 2.0, 'maximum multiplicative lighten/darken')

flags.DEFINE_float('channel_add_min', -0.15, 'minimum per channel additive lighten/darken')
flags.DEFINE_float('channel_add_max', 0.15, 'maximum per channel additive lighten/darken')

flags.DEFINE_float('channel_mult_min', 0.7, 'minimum per channel multiplicative lighten/darken')
flags.DEFINE_float('channel_mult_max', 1.3, 'maximum per channel multiplicative lighten/darken')

flags.DEFINE_float('stddev', 0.1, 'stddev for gaussian noise')


cfg = tf.app.flags.FLAGS
