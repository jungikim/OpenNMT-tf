PYTHONPATH=. python2 opennmt/bin/build_vocab.py --save_vocab ASLLRP-ASLLVD-OpenNMT-tf/ASL.vocab ASLLRP-ASLLVD-all-VOCAB.txt

#PYTHONPATH=. python opennmt/bin/video_to_tfrecord.py --vidList ASLLRP-ASLLVD-test.file --out ASLLRP-ASLLVD-OpenNMT-tf/test

PYTHONPATH=. python opennmt/bin/video_to_tfrecord.py --vidList ASLLRP-ASLLVD-OpenNMT-tf-data/ASLLRP-ASLLVD-test.file --out ASLLRP-ASLLVD-OpenNMT-tf-data/test


for i in test valid train ; do PYTHONPATH=. python opennmt/bin/video_to_tfrecord.py --vidList ASLLRP-ASLLVD-OpenNMT-tf-data/ASLLRP-ASLLVD-${i}.file --out ASLLRP-ASLLVD-OpenNMT-tf-data/${i} > ASLLRP-ASLLVD-OpenNMT-tf-data/${i}.log 2>&1 ; done &
