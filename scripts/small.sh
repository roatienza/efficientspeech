python3 tts_demo.py --checkpoint checkpoints/small_v2.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3

python3 tts_demo.py --checkpoint checkpoints/small_v2_tag.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3  --preprocess-config config/isip-preprocess.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --out-folder small_v2_eng  --devices 4 --head 1 --reduction 2 --expansion 1 --kernel-size 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --out-folder tiny_v2_tag  --reduction 4 --head 1 --expansion 1 --kernel-size 3 --n-blocks 2 --devices 4  --preprocess-config config/isip-preprocess.yaml
