PYTHONPATH=. python2 opennmt/bin/main.py --config config/data/librispeech.yml config/asr_transformer.yml --model_type TransformerASR --num_gpus 4 train > asr_librispeech_train.log 2>&1 &
