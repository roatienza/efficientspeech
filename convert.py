'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza, 2023
Apache 2.0 License

Usage:
    python3 convert.py --checkpoint tiny_eng_266k.ckpt --onnx tiny_eng_266k.onnx
'''

import torch
import yaml
from model import EfficientSpeech
from utils.tools import get_args

# main routine
if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)

    model = EfficientSpeech(preprocess_config=preprocess_config, infer_device=args.infer_device)
 
    model = model.load_from_checkpoint(args.checkpoint,
                                       preprocess_config=preprocess_config,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay,
                                       max_epochs=args.max_epochs,
                                       depth=args.depth,
                                       n_blocks=args.n_blocks,
                                       block_depth=args.block_depth,
                                       reduction=args.reduction,
                                       head=args.head,
                                       embed_dim=args.embed_dim,
                                       kernel_size=args.kernel_size,
                                       decoder_kernel_size=args.decoder_kernel_size,
                                       expansion=args.expansion,
                                       hifigan_checkpoint=args.hifigan_checkpoint,
                                       infer_device=args.infer_device,
                                       verbose=args.verbose)
    model = model.to(args.infer_device)
    # not needed but here it is
    model.eval()

    if args.onnx is not None:
        phoneme = torch.randint(low=70, high=146, size=(1,args.onnx_insize)).int()
        print("Input shape: ", phoneme.shape)
        sample_input = [{"phoneme": phoneme}, False]
        print("Converting to ONNX ...", args.onnx)
        
        # https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
        # or use model.to_onnx
        #model.to_onnx(args.onnx, sample_input, input_names="phoneme") #, export_params=True)
        torch.onnx.export(model, sample_input, args.onnx,
                            opset_version=args.onnx_opset, do_constant_folding=True,
                            input_names=["inputs"], output_names=["outputs"],
                            dynamic_axes={
                                "inputs": {1: "phoneme"},
                                "outputs": {1: "wav"} #ideally, this works but repeat_interleave is fixed
                            })
    elif args.jit is not None:
        with torch.no_grad():
            print("Converting to JIT ...", args.jit)
            #model.to_jit()
            script = model.to_torchscript()
            torch.jit.save(script, args.jit)
