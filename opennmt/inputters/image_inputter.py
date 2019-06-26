"""Define an image inputter for kaffe-tensorflow models."""

import abc
import six
import numpy as np

import tensorflow as tf

from opennmt.inputters.inputter import Inputter
from opennmt.utils.misc import count_lines


@six.add_metaclass(abc.ABCMeta)
class KaffeImageInputter(Inputter):
  """An inputter that processes images according to kaffe-tensorflow."""

  def __init__(self, modelname, dtype=tf.float32):
    super(KaffeImageInputter, self).__init__(dtype=dtype)
    self.modelname = modelname
     

  def get_length(self, data):
    return data["length"]

  def make_dataset(self, data_file):
    return tf.data.TextLineDataset(data_file)

  def get_dataset_size(self, data_file):
    return count_lines(data_file)

  def _process(self, data):
    """Tokenizes raw text."""
    data = super(KaffeImageInputter, self)._process(data)
    if "tokens" not in data:
      text = data["raw"]
      tokens = self.tokenizer.tokenize(text)
      length = tf.shape(tokens)[0]
      data = self.set_data_field(data, "tokens", tokens)
      data = self.set_data_field(data, "length", length)
    return data

  def _get_serving_input(self):
    receiver_tensors = {
        "tokens": tf.placeholder(tf.string, shape=(None, None)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }
    features = receiver_tensors.copy()
    features["ids"] = self.vocabulary.lookup(features["tokens"])
    return receiver_tensors, features

  def _process(self, data):

    data = super(AudioInputter, self)._process(data)

    features = tf.parse_single_example(data["raw"], features={
        "waveforms": tf.VarLenFeature(tf.float32)
    })

    waveforms = features["waveforms"].values
    waveforms = tf.expand_dims(waveforms, 0)
 
    mel_fbanks = common_audio.compute_mel_filterbank_features(
                                waveforms,
                                sample_rate=self.audio_sample_rate,
                                dither=self.audio_dither,
                                preemphasis=self.audio_preemphasis,
                                frame_length=self.audio_frame_length,
                                frame_step=self.audio_frame_step,
                                lower_edge_hertz=self.audio_lower_edge_hertz,
                                upper_edge_hertz=self.audio_upper_edge_hertz,
                                num_mel_bins=self.audio_num_mel_bins,
                                apply_mask=False)

    if self.audio_add_delta_deltas:
      mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)

    fbank_size = common_layers.shape_list(mel_fbanks)
    assert fbank_size[0] == 1

    # This replaces CMVN estimation on data
    var_epsilon = 1e-09
    mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
    variance = tf.reduce_mean(tf.square(mel_fbanks - mean),
                              keepdims=True, axis=1)
    mel_fbanks = (mel_fbanks - mean) * tf.rsqrt(variance + var_epsilon)

    # flatten frequency and channels (delta, deltadelta), 80 x 3
    tensor = tf.concat([
      tf.reshape(mel_fbanks, [fbank_size[1], fbank_size[2] * fbank_size[3]]),
      tf.zeros((self.num_zeropad_frames, fbank_size[2] * fbank_size[3]))], 0)

    data["tensor"] = tensor
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

