"""image to TFRecord converter.

The script takes a file that contains list of image files and optionally the indexed target text
to write aligned source and target data.
"""

from __future__ import print_function

import argparse
import io
import numpy as np
import tensorflow as tf

import opennmt.inputters.image_inputter as image_inputter

def image_to_records(imagefile_list, out_prefix, dtype=np.float32):
  """Converts img dataset to TFRecords."""
  record_writer = tf.python_io.TFRecordWriter(out_prefix + ".tfrecords")
  count = 0
  maxLen = 0
  with io.open(imagefile_list, encoding="utf-8") as image_files:
    for line in image_files:
      line = line.strip()
      img = image_inputter.get_image(line)
      image_inputter.write_sequence_record(img, record_writer)
      count += 1

  record_writer.close()
  print("Saved {} records.".format(count))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--imgList", required=True,
                      help="Text file with list of image files (jpg, png)")
  parser.add_argument("--out", required=True,
                      help="Output files prefix (will be suffixed by .tfrecords and .txt).")
  parser.add_argument("--dtype", default="float32",
                      help="Vector dtype")
  args = parser.parse_args()
  dtype = np.dtype(args.dtype)

  image_to_records(args.imgList, args.out, dtype=dtype)

if __name__ == "__main__":
  main()
