Demo:
python3 demo.py --checkpoint checkpoints/base_eng.ckpt --accelerator cpu --infer-device cpu

Train:
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --out-folder base_v2_eng  --devices 4

Benchmark:
python3 demo.py --checkpoint checkpoints/base_eng.ckpt --accelerator cuda --infer-device cuda    --text "the quick brown fox jumps over the lazy dog" --benchmark
