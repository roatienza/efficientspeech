'''
Torch model to onnx or jit converter

Conver a torch model:
    (Tagalog)
    ONNX:
      python3 convert.py --checkpoint checkpoints/tiny_v2_tag_attn.ckpt --accelerator cpu --infer-device cpu \
          --head 1 --reduction 4 --expansion 1 --kernel-size 3  --n-blocks 2 --preprocess-config config/isip-preprocess.yaml \
          --onnx checkpoints/tiny_v2_tag_attn.onnx > log.txt
'''

import torch
import yaml
from model import EfficientFSModule
from utils.tools import get_args

# main routine
if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)

    pl_module = EfficientFSModule(preprocess_config=preprocess_config, infer_device=args.infer_device)
 
    pl_module = pl_module.load_from_checkpoint(args.checkpoint, preprocess_config=preprocess_config,
                                               lr=args.lr, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                               depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                               reduction=args.reduction, head=args.head,
                                               embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                               decoder_kernel_size=args.decoder_kernel_size,
                                               expansion=args.expansion, 
                                               hifigan_checkpoint=args.hifigan_checkpoint,
                                               infer_device=args.infer_device, 
                                               verbose=args.verbose)
    pl_module.eval()

    if args.onnx is not None:
        phoneme = torch.randint(low=150, high=196, size=(1,64)).long()
        print("Input shape: ", phoneme.shape)
        sample_input = {"phoneme": phoneme, }
        print("Converting to ONNX ...", args.onnx)
        
        with torch.no_grad():
            # https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
            # or use pl_module.to_onnx
            torch.onnx.export(pl_module, sample_input, args.onnx, export_params=True,
                              opset_version=12, do_constant_folding=True, verbose=True,
                              input_names=["inputs"], output_names=["outputs"],
                              dynamic_axes={
                                  "inputs": {1: "phoneme"},
                                  "outputs": {1: "wav"}
                              })
    elif args.jit is not None:
        with torch.no_grad():
            print("Converting to JIT ...", args.jit)
            #pl_module.to_jit()
            script = pl_module.to_torchscript()
            torch.jit.save(script, args.jit)