CHECKPOINT_STEP=$1

if [ -z "$CHECKPOINT_STEP" ] ; then
  echo "PLEASE SPECIFY CHECKPOINT STEP"
  exit 1
fi

if [ ! -f ASLLRP-ASLLVD-OpenNMT-tf-model/model.ckpt-${CHECKPOINT_STEP}.index ] ; then
  echo "Checkpoint ASLLRP-ASLLVD-OpenNMT-tf-model/model.ckpt-${CHECKPOINT_STEP} does not exist"
  exit 1
fi

CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python2 opennmt/bin/main.py --config config/data/asllvd.yml config/VideoCTCTagger.yml --model_type VideoCTCTagger --features_file ASLLRP-ASLLVD-OpenNMT-tf-data/ASLLRP-ASLLVD-test.tfrecords --checkpoint_path ASLLRP-ASLLVD-OpenNMT-tf-model/model.ckpt-${CHECKPOINT_STEP} --num_gpus 0 infer > aslr_asllvd_03_infer_test_ckpt-${CHECKPOINT_STEP}.txt 2> aslr_asllvd_03_infer_test_ckpt-${CHECKPOINT_STEP}.log &




