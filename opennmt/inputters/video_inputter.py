"""Define an video inputter for kaffe-tensorflow models."""

import abc
import six
import numpy as np
import os

import tensorflow as tf

import cv2

from opennmt.inputters.inputter import Inputter
from opennmt.utils import compat

from __builtin__ import False

@six.add_metaclass(abc.ABCMeta)
class VideoInputter(Inputter):
  """An inputter that processes Videos where each frame is processed by the AlexNet model from kaffe-tensorflow."""

  def __init__(self, dtype=tf.float32):
    super(VideoInputter, self).__init__(dtype=dtype)

  def get_length(self, data):
    return data["length"]

#  def make_dataset(self, data_file, training=None):
#    return tf.data.TextLineDataset(data_file)

#  def get_dataset_size(self, data_file):
#    return count_lines(data_file)

  def make_dataset(self, data_file, training=None):
    return tf.data.TFRecordDataset(data_file)

  def get_dataset_size(self, data_file):
    return sum(1 for _ in tf.python_io.tf_record_iterator(data_file))

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "tensor" in features:
      return features
    tf_parse_example = compat.tf_compat(v2="io.parse_single_example", v1="parse_single_example")
    tf_var_len_feature = compat.tf_compat(v2="io.VarLenFeature", v1="VarLenFeature")
    example = tf_parse_example(element, features={
        "shape": tf_var_len_feature(tf.int64),
        "values": tf_var_len_feature(tf.float32)
    })

    values = example["values"].values
    shape = tf.cast(example["shape"].values, tf.int32)
#    tensor = tf.reshape(values, shape)

    tensor = tf.reshape(values, [shape[0], shape[1] * shape[2] * shape[3]])
    
    features["length"] = tf.shape(tensor)[0]
    features["tensor"] = tf.cast(tensor, self.dtype)
    return features

  def make_inputs(self, features, training=None):
    outputs = features["tensor"]
#    if training and self.dropout > 0:
#      outputs = tf.keras.layers.Dropout(self.dropout)(outputs, training=training)
    return outputs

#   def _get_serving_input(self):
#     receiver_tensors = {
#         "tokens": tf.placeholder(tf.string, shape=(None, None)),
#         "length": tf.placeholder(tf.int32, shape=(None,))
#     }
#     features = receiver_tensors.copy()
#     features["ids"] = self.vocabulary.lookup(features["tokens"])
#     return receiver_tensors, features


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

def get_video(videoF):
  #alexnet_spec
  alex_net_scale_size = 256
  alex_net_crop_size = 227
  alex_net_isotropic = False
  #alex_net_channels = 3
  alex_net_mean = np.array([104., 117., 124.])
  #alex_net_expects_bgr = True
  
  """Yield images and their frame number from a video file."""
  vidcap = cv2.VideoCapture(videoF)  
  len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  imgList = []
  
  for _ in range(len):
    success, image = vidcap.read()
    # image is already bgr with 3 channels
    if not success:
      break
    image = kaffe_imagenet_process_image(
                              img=image,
                              scale=alex_net_scale_size,
                              isotropic=alex_net_isotropic,
                              crop=alex_net_crop_size,
                              mean=alex_net_mean)
    imgList.append(image)

  vid=np.stack(imgList, axis=0)
  return vid


def write_sequence_record(vid, writer):
  """Writes a vector as a TFRecord.
  Args:
    vid: A list of float numbers.
    writer: A ``tf.python_io.TFRecordWriter``.
  """
  shape = list(vid.shape)
  values = vid.flatten().tolist()

  data = tf.train.Example(features=tf.train.Features(feature={
      "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=shape)),
      "values": tf.train.Feature(float_list=tf.train.FloatList(value=values))
  }))

  writer.write(data.SerializeToString())
