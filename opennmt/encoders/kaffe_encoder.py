"""Define convolution-based encoders."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import PositionEmbedder

from kaffe.imagenet import models

class KaffeEncoder(Encoder):
  """An encoder that applies various pre-trained Caffe CNN models (https://github.com/BVLC/caffe/tree/master/models/)
    converted using Kaffe-tensorflow (https://github.com/ethereon/caffe-tensorflow/)

  """

  def __init__(self, modelname):
    """Initializes the parameters of the encoder.

    Args:
      modelname: 
      modelpath: 
    """

    self.modelname = modelname

    self._model_dict = { 'AlexNet': (models.AlexNet, 'fc7'),
                         'CaffeNet': (models.CaffeNet, 'fc7'),
                         'VGG16': (models.VGG16, 'fc7'),
                         'GoogleNet': (models.GoogleNet, 'pool5_7x7_s1'),
                         'NiN': (models.NiN, 'pool4'),
                         'ResNet50': (models.ResNet50, 'pool5'),
                         'ResNet101': (models.ResNet101, 'pool5'),
                         'ResNet152': (models.ResNet152, 'pool5')}

    if self.modelname in self._model_dict: 
        self.model, self._output_layer_name = self.modelDict[self.modelname]
    else:
        raise Exception('Unknown model name')


  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):

#     # squeeze sequence dimension
#     inputs = tf.squeeze(inputs, [1])

    with tf.variable_scope('kaffe'):
        self.net = self.model({'data': inputs, 'use_dropout': mode == tf.estimator.ModeKeys.TRAIN})

    encoder_output = self.net.layers[self._output_layer_name]
    # add back sequence dimension    
#     encoder_output = tf.reshape(encoder_output, [encoder_output[0], 1, encoder_output[1]])
    encoder_state = encoder_output

    return (encoder_output, encoder_state, sequence_length)


  def load(self, modelpath, session):
    self.net.load(modelpath, session)
