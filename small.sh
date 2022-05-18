python3 tts_demo.py --checkpoint checkpoints/small_v2.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3

python3 tts_demo.py --checkpoint checkpoints/small_v2_tag.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3  --preprocess-config config/isip-preprocess.yaml
