"""audio to TFRecord converter.

The script takes a file that contains list of audio files and optionally the indexed target text
to write aligned source and target data.
"""

from __future__ import print_function

import argparse
import io
import numpy as np
import tensorflow as tf

import opennmt.inputters.audio_inputter as audio_inputter

def audio_to_records(audiofile_list, out_prefix, dtype=np.float32):
  """Converts audio dataset to TFRecords."""
  record_writer = tf.python_io.TFRecordWriter(out_prefix + ".tfrecords")
  count = 0

  maxLen = 0

  with io.open(audiofile_list, encoding="utf-8") as audio_files:
    for line in audio_files:
      line = line.strip()
      waveform = audio_inputter.get_waveform(line)
      audio_inputter.write_sequence_record(waveform, record_writer)
      count += 1
      if len(waveform)/80 > maxLen:
        maxLen = len(waveform)/80

  record_writer.close()
  print("Saved {} records.".format(count))
  print("MaxLen: {}".format(maxLen))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--audioList", required=True,
                      help="Text file with list of audio files (wav, mp3)")
  parser.add_argument("--out", required=True,
                      help="Output files prefix (will be suffixed by .tfrecords and .txt).")
  parser.add_argument("--dtype", default="float32",
                      help="Vector dtype")
  args = parser.parse_args()
  dtype = np.dtype(args.dtype)

  audio_to_records(args.audioList, args.out, dtype=dtype)

if __name__ == "__main__":
  main()
