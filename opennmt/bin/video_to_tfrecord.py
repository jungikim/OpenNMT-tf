"""video to TFRecord converter.

The script takes a file that contains list of video files and optionally the indexed target text
to write aligned source and target data.
"""

from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import io
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

import opennmt.inputters.video_inputter as video_inputter

def video_to_records(videofile_list, out_prefix, dtype=np.float32):

  with tf.device('/cpu:0'):
      """Converts vid dataset to TFRecords."""
      record_writer = tf.python_io.TFRecordWriter(out_prefix + ".tfrecords")
      count = 0
      maxLen = 0
      with io.open(videofile_list, encoding="utf-8") as video_files:
        for line in video_files:
          line = line.strip()
          print(line)
          vid = video_inputter.get_video(line)
          video_inputter.write_sequence_record(vid, record_writer)
          count += 1
    
      record_writer.close()
      print("Saved {} records.".format(count))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vidList", required=True,
                      help="Text file with list of video files")
  parser.add_argument("--out", required=True,
                      help="Output files prefix (will be suffixed by .tfrecords and .txt).")
  parser.add_argument("--dtype", default="float32",
                      help="Vector dtype")
  args = parser.parse_args()
  dtype = np.dtype(args.dtype)

  video_to_records(args.vidList, args.out, dtype=dtype)

if __name__ == "__main__":
  main()
