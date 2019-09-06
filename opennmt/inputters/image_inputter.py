"""Define an image inputter for kaffe-tensorflow models."""

import abc
import six
import numpy as np
import os

import tensorflow as tf

from opennmt.inputters.inputter import Inputter
from __builtin__ import False

@six.add_metaclass(abc.ABCMeta)
class ImageInputter(Inputter):
  """An inputter that processes images for the AlexNet from kaffe-tensorflow."""

  def __init__(self, modelname, dtype=tf.float32):
    super(ImageInputter, self).__init__(dtype=dtype)
    self.modelname = modelname

  def get_length(self, data):
    return data["length"]

  def make_dataset(self, data_file):
    return tf.data.TFRecordDataset(data_file)

  def get_dataset_size(self, data_file):
    return sum(1 for _ in tf.python_io.tf_record_iterator(data_file))

#   def _get_serving_input(self):
#     receiver_tensors = {
#         "tokens": tf.placeholder(tf.string, shape=(None, None)),
#         "length": tf.placeholder(tf.int32, shape=(None,))
#     }
#     features = receiver_tensors.copy()
#     features["ids"] = self.vocabulary.lookup(features["tokens"])
#     return receiver_tensors, features

  def make_features(self, data):

    data = super(ImageInputter, self)._process(data)

    features = tf.parse_single_example(data["raw"], features={
        "shape": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32)
    })

    shape = tf.cast(features["shape"].values, tf.int32)
    tensor = features["values"].values
    tensor = tf.reshape(tensor, shape)
    length = tf.shape(tensor)[0]

    data = self.set_data_field(data, "tensor", tensor)
    data = self.set_data_field(data, "length", length)

    return data

  def _transform_data(self, data, mode):
    return self.transform(data["tensor"], mode)

  def transform(self, inputs, mode):
    outputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs


def kaffe_imagenet_load_image(image_path, channels, expects_bgr):
  file_data = tf.read_file(image_path)
  img = tf.image.decode_image(file_data, channels=channels)
  if expects_bgr:
    # Convert from RGB channel ordering to BGR
    # This matches, for instance, how OpenCV orders the channels.
    img = tf.reverse(img, [False, False, True])
  return img

def kaffe_imagenet_process_image(img, scale, isotropic, crop, mean):
  '''Crops, scales, and normalizes the given image.
  scale : The image wil be first scaled to this size.
          If isotropic is true, the smaller side is rescaled to this,
          preserving the aspect ratio.
  crop  : After scaling, a central crop of this size is taken.
  mean  : Subtracted from the image
  '''
  # Rescale
  if isotropic:
    img_shape = tf.to_float(tf.shape(img)[:2])
    min_length = tf.minimum(img_shape[0], img_shape[1])
    new_shape = tf.to_int32((scale / min_length) * img_shape)
  else:
    new_shape = tf.stack([scale, scale])
  img = tf.image.resize_images(img, [new_shape[0], new_shape[1]])
  # Center crop
  # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
  # See: https://github.com/tensorflow/tensorflow/issues/521
  offset = (new_shape - crop) / 2
  img = tf.slice(img, begin=tf.stack([offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
  # Mean subtraction
  return tf.to_float(img) - mean


def get_image(imageF):
  #alexnet_spec
  alex_net_scale_size = 256
  alex_net_crop_size = 227
  alex_net_isotropic = False
  alex_net_channels = 3
  alex_net_mean = np.array([104., 117., 124.])
  alex_net_expects_bgr = True

  img = kaffe_imagenet_load_image(imageF, alex_net_channels,
                                          alex_net_channels,
                                          alex_net_expects_bgr)
  # img dim: [h,w,c]
  img = kaffe_imagenet_process_image(img=img,
                                scale=alex_net_scale_size,
                                isotropic=alex_net_isotropic,
                                crop=alex_net_crop_size,
                                mean=alex_net_mean)
  return img

def write_sequence_record(tensor, writer):
  shape = list(tensor.shape)
  values = tf.reshape(tensor,[-1])

  example = tf.train.Example(features=tf.train.Features(feature={
      "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=shape)),
      "values": tf.train.Feature(float_list=tf.train.FloatList(value=values))
  }))

  writer.write(example.SerializeToString())

