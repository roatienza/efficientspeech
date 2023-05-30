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

    model = EfficientSpeech(preprocess_config=preprocess_config)
    model = model.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
    model = model.to(args.infer_device)

    if args.onnx is not None:
        phoneme = torch.randint(low=70, high=146, size=(1,args.onnx_insize)).int().to(args.infer_device)
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
                              # ideally, this works but repeat_interleave is fixed
                              "outputs": {0: "wav", 1: "lengths", 2: "duration"}
                          })
    elif args.jit is not None:
        with torch.no_grad():
            print("Converting to JIT ...", args.jit)
            #model.to_jit()
            script = model.to_torchscript()
            torch.jit.save(script, args.jit)
