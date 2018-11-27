for i in librispeech_data/*-flacs.txt ; do
  echo $i;
  PYTHONPATH=. python2 opennmt/bin/audio_to_tfrecord.py --audioList ${i} --out ${i%%.txt} ;
done

# Expected output
#  librispeech_data/dev-all-flacs.txt
#  Saved 5567 records.
#  librispeech_data/test-clean-flacs.txt
#  Saved 2620 records.
#  librispeech_data/train-all-flacs.txt
#  Saved 281241 records.

# ls -hal librispeech_data/*.tfrecords
#-rw-r--r--  1 USER GROUP 1.2G Sep 26 16:43 test-clean-flacs.tfrecords
#-rw-r--r--  1 USER GROUP 2.3G Sep 26 16:42 dev-all-flacs.tfrecords
#-rw-r--r--  1 USER GROUP 207G Sep 26 19:54 train-all-flacs.tfrecords


cat librispeech_data/*-transcripts.txt > librispeech_data/transcripts.txt
PYTHONPATH=. python2 opennmt/bin/build_vocab.py --save_vocab librispeech_data/transcripts.vocab librispeech_data/transcripts.txt

