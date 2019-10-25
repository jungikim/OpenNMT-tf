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

  def __init__(self, dropout=0.0, frameskip=0, dtype=tf.float32):
    super(VideoInputter, self).__init__(dtype=dtype)
    self.dropout=dropout
    self.frameskip=frameskip

  def get_length(self, data):
    return data["length"]

  def make_dataset(self, data_file, training=None):
    return tf.data.TFRecordDataset(data_file)

  def get_dataset_size(self, data_file):
    return sum(1 for _ in tf.python_io.tf_record_iterator(data_file))

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "tensor" in features:
      return features

    sequence_features = {
      'frames': tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    features, sequence_features = tf.parse_single_sequence_example(element, context_features=None,
                                                             sequence_features=sequence_features)

    images = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), sequence_features['frames'], dtype=tf.uint8)

    shape = tf.shape(images)
    tensor = tf.reshape(images, [shape[0], shape[1] * shape[2] * shape[3]])

    if training and self.frameskip > 0:
      featlen = tf.reshape(tf.shape(tensor)[0],[])
      toSkip = tf.random_uniform(shape=[], minval=1, maxval=self.frameskip, dtype=tf.int32)
#      desiredNumFrames = tf.floor(tf.compat.v1.div(featlen, toSkip))
      beginOffset = tf.math.minimum(featlen, tf.random_uniform(shape=[], minval=0, maxval=toSkip, dtype=tf.int32))
      offsets = tf.compat.v1.range(beginOffset, featlen, delta=toSkip)
      selectedFrames = tf.map_fn(lambda i: tensor[i], offsets, dtype=(self.dtype))
      features["length"] = tf.shape(selectedFrames)[0]
      features["tensor"] = tf.cast(selectedFrames, self.dtype)
    else:
      features["length"] = tf.shape(tensor)[0]
      features["tensor"] = tf.cast(tensor, self.dtype)

    return features

  def make_inputs(self, features, training=None):
    outputs = features["tensor"]

    if training and self.dropout > 0:
      outputs = tf.keras.layers.Dropout(self.dropout)(outputs, training=training)
    return outputs

  def get_receiver_tensors(self):
    return {
        "tensor": tf.placeholder(self.dtype, shape=(None, None, 227*227*3)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

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
#  cropped = tf.image.resize_image_with_crop_or_pad(img, crop, crop)
  # Mean subtraction
  return (img - mean).numpy()

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

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imgList.append(tf.compat.as_bytes(cv2.imencode(".jpg", image)[1].tobytes()))

  return imgList


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def write_sequence_record(vid, writer):
  """Writes a vector as a TFRecord.
  Args:
    vid: A list of float numbers.
    writer: A ``tf.python_io.TFRecordWriter``.
  """

  feature_list = {'frames': _bytes_feature_list(vid)}
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  example = tf.train.SequenceExample(feature_lists=feature_lists, context=None)
  writer.write(example.SerializeToString())
