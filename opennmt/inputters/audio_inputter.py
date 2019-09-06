"""Define audio inputters."""

import abc
import six
import numpy as np

import tensorflow as tf

from opennmt.inputters.inputter import Inputter
from opennmt.utils.misc import count_lines

import t2t_common_audio as common_audio
import t2t_common_layers as common_layers
from t2t_audio_encoder import AudioEncoder

@six.add_metaclass(abc.ABCMeta)
class AudioInputter(Inputter):
  """An inputter that processes audio waveform signal."""

  def __init__(self,
               #
               audio_sample_rate = 16000,
               audio_preemphasis = 0.97,
               audio_dither = 1.0 / np.iinfo(np.int16).max,
               audio_frame_length = 25.0,
               audio_frame_step = 10.0,
               audio_lower_edge_hertz = 20.0,
               audio_upper_edge_hertz = 8000.0,
               audio_num_mel_bins = 80,
               audio_add_delta_deltas = True,
               num_zeropad_frames = 250,
               #
               dropout=0.0,
               dtype=tf.float32):
    """Initializes the parameters of the word embedder.
    Args:
      dropout: The probability to drop units in the audio signal.
      dtype: The embedding type.
    Raises:
      ValueError: if neither :obj:`embedding_size` nor :obj:`embedding_file_key`
        are set.
    See Also:
      The :meth:`opennmt.inputters.text_inputter.load_pretrained_embeddings`
      function for details about the pretrained embedding format and behavior.
    """
    super(AudioInputter, self).__init__(dtype=dtype)

    self.audio_sample_rate = audio_sample_rate 
    self.audio_preemphasis = audio_preemphasis 
    self.audio_dither = audio_dither 
    self.audio_frame_length = audio_frame_length 
    self.audio_frame_step = audio_frame_step 
    self.audio_lower_edge_hertz = audio_lower_edge_hertz 
    self.audio_upper_edge_hertz = audio_upper_edge_hertz 
    self.audio_num_mel_bins = audio_num_mel_bins 
    self.audio_add_delta_deltas = audio_add_delta_deltas
    self.num_zeropad_frames = num_zeropad_frames 

    self.dropout = dropout

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
# 
#     features = receiver_tensors.copy()
#     features["ids"] = self.vocabulary.lookup(features["tokens"])
# 
#     return receiver_tensors, features

  def make_features(self, data):

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

def get_waveform(inputF):
  return AudioEncoder().encode(inputF)

def write_sequence_record(waveform, writer):
  """Writes a vector as a TFRecord.
  Args:
    waveform: A list of float numbers.
    writer: A ``tf.python_io.TFRecordWriter``.
  """
  data = tf.train.Example(features=tf.train.Features(feature={
      "waveforms": tf.train.Feature(float_list=tf.train.FloatList(value=waveform))
  }))

  writer.write(data.SerializeToString())

