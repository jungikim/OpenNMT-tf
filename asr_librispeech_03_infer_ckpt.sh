
CHECKPOINT_STEP=$1

if [ -z "$CHECKPOINT_STEP" ] ; then
  echo "PLEASE SPECIFY CHECKPOINT STEP"
  exit 1
fi

if [ ! -f librispeech_model/model.ckpt-${CHECKPOINT_STEP} ] ; then
  echo "Checkpoint librispeech_model/model.ckpt-${CHECKPOINT_STEP} does not exist"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python2 opennmt/bin/main.py --config config/data/librispeech.yml config/asr_transformer.yml --model_type TransformerASR --features_file librispeech_data/test-clean-flacs.tfrecords --checkpoint_path librispeech_model/model.ckpt-${CHECKPOINT_STEP} --num_gpus 1 infer asr_librispeech_infer_test-clean_ckpt-${CHECKPOINT_STEP}.txt 2> asr_librispeech_infer_test-clean_ckpt-${CHECKPOINT_STEP}.log &
