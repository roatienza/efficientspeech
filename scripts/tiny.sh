Demo:
python3 demo.py --checkpoint checkpoints/tiny_eng.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 4 --expansion 1 --kernel-size 3 --n-blocks 2

python3 demo.py --checkpoint checkpoints/tiny_tag.ckpt --accelerator cpu --infer-device cpu --head 1 --reduction 4 --expansion 1 --kernel-size 3  --n-blocks 2 --preprocess-config config/isip-preprocess.yaml

Train: (4GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --out-folder tiny_v2_eng --reduction 4 --head 1 --expansion 1 --kernel-size 3 --n-blocks 2 --devices 4

Benchmark:
python3 demo.py --checkpoint checkpoints/tiny_eng.ckpt --accelerator cuda --infer-device cuda --head 1 --reduction 4 --expansion 1 --kernel-size 3 --n-blocks 2 --text "the quick brown fox jumps over the lazy dog" --benchmark
