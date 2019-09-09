PYTHONPATH=. python2 opennmt/bin/main.py --config config/data/asllvd.yml config/VideoCTCTagger.yml --model_type VideoCTCTagger --num_gpus 4 train > aslr_asllvd_02_train.log 2>&1 &
