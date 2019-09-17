"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

from opennmt.utils.misc import merge_dict


class ListenAttendSpell(onmt.models.SequenceToSequence):
  """Defines a model similar to the "Listen, Attend and Spell" model described
  in https://arxiv.org/abs/1508.01211.
  """
  def __init__(self):
    super(ListenAttendSpell, self).__init__(
        source_inputter=onmt.inputters.SequenceRecordInputter(),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_vocabulary",
            embedding_size=50),
        encoder=onmt.encoders.PyramidalRNNEncoder(
            num_layers=3,
            num_units=512,
            reduction_factor=2,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3),
        decoder=onmt.decoders.MultiAttentionalRNNDecoder(
            num_layers=3,
            num_units=512,
            attention_layers=[0],
            attention_mechanism_class=tf.contrib.seq2seq.LuongMonotonicAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

  def auto_config(self, num_devices=1):
    config = super(ListenAttendSpell, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "GradientDescentOptimizer",
            "learning_rate": 0.2,
            "clip_gradients": 10.0,
            "scheduled_sampling_type": "constant",
            "scheduled_sampling_read_probability": 0.9
        },
        "train": {
            "batch_size": 32,
            "bucket_width": 15,
            "maximum_features_length": 2450,
            "maximum_labels_length": 330
        }
    })

class _RNNBase(onmt.models.SequenceToSequence):
  """Base class for RNN based NMT models."""
  def __init__(self, *args, **kwargs):
    super(_RNNBase, self).__init__(*args, **kwargs)

  def auto_config(self, num_devices=1):
    config = super(_RNNBase, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "AdamOptimizer",
            "learning_rate": 0.0002,
            "param_init": 0.1,
            "clip_gradients": 5.0
        },
        "train": {
            "batch_size": 64,
            "maximum_features_length": 80,
            "maximum_labels_length": 80
        }
    })

class ListenAttendSpellASR(onmt.models.SequenceToSequence):
  """Defines a model similar to the "Listen, Attend and Spell" model described
  in https://arxiv.org/abs/1508.01211.
  """
  def __init__(self):
    super(ListenAttendSpellASR, self).__init__(
        source_inputter=onmt.inputters.AudioInputter(),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=50),
        encoder=onmt.encoders.PyramidalRNNEncoder(
            num_layers=3,
            num_units=512,
            reduction_factor=2,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3),
        decoder=onmt.decoders.MultiAttentionalRNNDecoder(
            num_layers=3,
            num_units=512,
            attention_layers=[0],
            attention_mechanism_class=tf.contrib.seq2seq.LuongMonotonicAttention,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class TransformerASR(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self):
    super(TransformerASR, self).__init__(
        source_inputter=onmt.inputters.AudioInputter(),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=64),
        num_layers=4,
        num_units=384,
        num_heads=2,
        ffn_inner_dim=1536,
        dropout=0.1,
        attention_dropout=0.2,
        relu_dropout=0.2)

class NMTBig(_RNNBase):
  """Defines a bidirectional LSTM encoder-decoder model."""
  def __init__(self):
    super(NMTBig, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        encoder=onmt.encoders.BidirectionalRNNEncoder(
            num_layers=4,
            num_units=1024,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=4,
            num_units=1024,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class NMTMedium(_RNNBase):
  """Defines a medium-sized bidirectional LSTM encoder-decoder model."""
  def __init__(self):
    super(NMTMedium, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        encoder=onmt.encoders.BidirectionalRNNEncoder(
            num_layers=4,
            num_units=512,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=4,
            num_units=512,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class NMTSmall(_RNNBase):
  """Defines a small unidirectional LSTM encoder-decoder model."""
  def __init__(self):
    super(NMTSmall, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        encoder=onmt.encoders.UnidirectionalRNNEncoder(
            num_layers=2,
            num_units=512,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=2,
            num_units=512,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class SeqTagger(onmt.models.SequenceTagger):
  """Defines a bidirectional LSTM-CNNs-CRF as described in https://arxiv.org/abs/1603.01354."""
  def __init__(self):
    # pylint: disable=bad-continuation
    super(SeqTagger, self).__init__(
        inputter=onmt.inputters.MixedInputter([
            onmt.inputters.WordEmbedder(
                vocabulary_file_key="words_vocabulary",
                embedding_size=None,
                embedding_file_key="words_embedding",
                trainable=True),
            onmt.inputters.CharConvEmbedder(
                vocabulary_file_key="chars_vocabulary",
                embedding_size=30,
                num_outputs=30,
                kernel_size=3,
                stride=1,
                dropout=0.5)],
            dropout=0.5),
        encoder=onmt.encoders.BidirectionalRNNEncoder(
            num_layers=1,
            num_units=400,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.5,
            residual_connections=False),
        labels_vocabulary_file_key="tags_vocabulary",
        crf_decoding=True)

  def auto_config(self, num_devices=1):
    config = super(SeqTagger, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "AdamOptimizer",
            "learning_rate": 0.001
        },
        "train": {
            "batch_size": 32
        }
    })

class VideoClassifier(onmt.models.SequenceClassifier):
  """  """
  def __init__(self):
    super(VideoClassifier, self).__init__(
        inputter=onmt.inputters.VideoInputter(),
        encoder= onmt.encoders.SequentialEncoder([
        onmt.encoders.KaffeEncoder('AlexNet'),
        onmt.encoders.BidirectionalRNNEncoder(
            num_layers=2,
            num_units=256,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False)
        ]),
        labels_vocabulary_file_key="tags_vocabulary")


class VideoCTCTagger(onmt.models.SequenceTagger):
  """Defines a """
  def __init__(self):
    # pylint: disable=bad-continuation
    super(VideoCTCTagger, self).__init__(
        inputter=onmt.inputters.VideoInputter(),
        encoder= onmt.encoders.SequentialEncoder([
        onmt.encoders.KaffeEncoder('AlexNet'),
        onmt.encoders.BidirectionalRNNEncoder(
            num_layers=2,
            num_units=256,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False)
        ]),

        labels_vocabulary_file_key="tags_vocabulary",
        ctc_decoding=True)

  def auto_config(self, num_devices=1):
    config = super(VideoCTCTagger, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "AdamOptimizer",
            "learning_rate": 0.001
        },
        "train": {
            "batch_size": 32
        }
    })

class VideoTransformer(onmt.models.SequenceToSequence):
  """Defines a """
  
  def __init__(self, dtype=tf.float32):
    # pylint: disable=bad-continuation
    position_encoder = onmt.layers.position.SinusoidalPositionEncoder()
    num_layers=3
    num_units=256
    num_heads=8
    ffn_inner_dim=1024
    dropout=0.1
    attention_dropout=0.1
    relu_dropout=0.1
    decoder_self_attention_type="scaled_dot"

    super(VideoTransformer, self).__init__(
        source_inputter=onmt.inputters.VideoInputter(),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        encoder= onmt.encoders.SequentialEncoder([
            onmt.encoders.KaffeEncoder('AlexNet'),
            onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
                                                num_layers,
                                                num_units=num_units,
                                                num_heads=num_heads,
                                                ffn_inner_dim=ffn_inner_dim,
                                                dropout=dropout,
                                                attention_dropout=attention_dropout,
                                                relu_dropout=relu_dropout,
                                                position_encoder=position_encoder)
        ]),
        decoder = onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
                                            num_layers,
                                            num_units=num_units,
                                            num_heads=num_heads,
                                            ffn_inner_dim=ffn_inner_dim,
                                            dropout=dropout,
                                            attention_dropout=attention_dropout,
                                            relu_dropout=relu_dropout,
                                            position_encoder=position_encoder,
                                            self_attention_type=decoder_self_attention_type)
        )

  def auto_config(self, num_devices=1):
    config = super(VideoTransformer, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "average_loss_in_time": True,
            "label_smoothing": 0.1,
            "optimizer": "LazyAdamOptimizer",
            "optimizer_params": {
                "beta1": 0.9,
                "beta2": 0.998
            },
            "learning_rate": 2.0,
            "decay_type": "noam_decay_v2",
            "decay_params": {
                "model_dim": self._num_units,
                "warmup_steps": 8000
            }
        },
        "train": {
            "effective_batch_size": 25000,
            "batch_size": 3072,
            "batch_type": "tokens",
            "maximum_features_length": 100,
            "maximum_labels_length": 100,
            "keep_checkpoint_max": 8,
            "average_last_checkpoints": 8
        }
    })

  def _initializer(self, params):
    return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)


class Transformer(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self, dtype=tf.float32):
    super(Transformer, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1)

class TransformerFP16(Transformer):
  """Defines a Transformer model that uses half-precision floating points."""
  def __init__(self):
    super(TransformerFP16, self).__init__(dtype=tf.float16)

class TransformerAAN(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762
  with cumulative average attention in the decoder as described in
  https://arxiv.org/abs/1805.00631."""
  def __init__(self):
    super(TransformerAAN, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,
        decoder_self_attention_type="average")

class TransformerBig(onmt.models.Transformer):
  """Defines a large Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self, dtype=tf.float32):
    super(TransformerBig, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=1024,
            dtype=dtype),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=1024,
            dtype=dtype),
        num_layers=6,
        num_units=1024,
        num_heads=16,
        ffn_inner_dim=4096,
        dropout=0.3,
        attention_dropout=0.1,
        relu_dropout=0.1)

class TransformerBigFP16(TransformerBig):
  """Defines a large Transformer model that uses half-precision floating points."""
  def __init__(self):
    super(TransformerBigFP16, self).__init__(dtype=tf.float16)
