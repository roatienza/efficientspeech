python3 tts_demo.py --checkpoint checkpoints/tiny_v2.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 4 --expansion 1 --kernel-size 3 --n-blocks 2

python3 tts_demo.py --checkpoint checkpoints/tiny_v2_tag.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 4 --expansion 1 --kernel-size 3  --n-blocks 2 --preprocess-config config/isip-preprocess.yaml
