'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023

Usage:
    Torch:
    python3 demo.py --checkpoint tiny_eng_266k.ckpt --infer-device cuda  --text "In additive color mixing, which is used for displays such as computer screens and televisions, the primary colors are red, green, and blue."  --wav-filename color.wav

    ONNX:
    python3 demo.py --checkpoint tiny_eng_266k.onnx --infer-device cuda  --text "In additive color mixing, which is used for displays such as computer screens and televisions, the primary colors are red, green, and blue."  --wav-filename color.wav
    
Additional dependencies for GUI:
    pip3 install pysimplegui
    pip3 install sounddevice 
'''

import torch
import yaml
import time
import numpy as np
import validators


from model import EfficientSpeech
from utils.tools import get_args, write_to_file
from synthesize import get_lexicon_and_g2p, text2phoneme

def tts(lexicon, g2p, preprocess_config, model, is_onnx, args, verbose=False):
    text = args.text.strip()
    text = text.replace('-', ' ')
    phoneme = np.array(
            [text2phoneme(lexicon, g2p, text, preprocess_config, verbose=args.verbose)], dtype=np.int32)
    start_time = time.time()
    if is_onnx:
        # onnx is 3.5x faster than pytorch models
        phoneme_len = phoneme.shape[1]
        
        text = text + 2*args.onnx_insize*'- '
        phoneme = np.array(
            [text2phoneme(lexicon, g2p, text, preprocess_config, verbose=args.verbose)], dtype=np.int32)
        
        # unfortunately, due to call to repeat_interleave(), dynamic axis is not supported
        # so, the input size must be fixed to args.onnx_insize=128 (can be configured)
        phoneme = phoneme[:, :args.onnx_insize]
        
        ort_inputs = {model.get_inputs()[0].name: phoneme}
        outputs = model.run(None, ort_inputs)
        wavs = outputs[0]
        hop_len = preprocess_config["preprocessing"]["stft"]["hop_length"]
        lengths = outputs[1]
        
        duration = outputs[2]
        orig_duration = int(np.sum(np.round(duration.squeeze())[:phoneme_len])) * hop_len
        
        # crude estimate of duration
        # orig_duration = int(lengths*phoneme_len/args.onnx_insize) * hop_len
        
        # truncate the wav file to the original duration
        wavs = wavs[:, :orig_duration]
        lengths = [orig_duration]
    else:
        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int().to(args.infer_device)
            wavs, lengths, _ = model({"phoneme": phoneme})
            wavs = wavs.cpu().numpy()
            lengths = lengths.cpu().numpy()
        
    elapsed_time = time.time() - start_time
    #if is_onnx:
    #    elapsed_time *= (wav.shape[0] / outputs[0].shape[1])
    wav = np.reshape(wavs, (-1, 1))

    message = f"Synthesis time: {elapsed_time:.2f} sec"
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wav_len = wav.shape[0] / sampling_rate
    message += f"\nVoice length: {wav_len:.2f} sec"
    real_time_factor = wav_len / elapsed_time
    message += f"\nReal time factor: {real_time_factor:.2f}"
    message += f"\nNote:\tFor benchmarking, load the model 1st, do a warmup run for 100x, then run the benchmark for 1000 iterations."
    message += f"\n\tGet the mean of 1000 runs. Use --iter N to run N iterations. eg N=100"
    if not args.play:
        write_to_file(wavs, preprocess_config, lengths=lengths, \
            wav_path=args.wav_path, filename=args.wav_filename)
    
    print(message)
    return wav, message, phoneme, wav_len, real_time_factor

if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
 
    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    is_onnx = False

    if validators.url(args.checkpoint):
        checkpoint = args.checkpoint.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    else:
        checkpoint = args.checkpoint


    if "onnx" in checkpoint:
        import onnxruntime
        import onnx

        onnx_model = onnx.load(checkpoint)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(checkpoint)
        model = ort_session
        is_onnx = True
    else:
        model = EfficientSpeech(preprocess_config=preprocess_config, 
                                infer_device=args.infer_device,
                                hifigan_checkpoint=args.hifigan_checkpoint,)

        model = model.load_from_checkpoint(checkpoint,
                                           infer_device=args.infer_device,
                                           map_location=torch.device('cpu'))
        

        model = model.to(args.infer_device)
        model.eval()
        
        # default number of threads is 128 on AMD
        # this is too high and causes the model to run slower
        # set it to a lower number eg --threads 24 
        # https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        if args.compile:
            model = torch.compile(model, mode="reduce-overhead", backend="inductor")
            
    if args.play:
        import sounddevice as sd
        sd.default.reset()
        sd.default.samplerate = sampling_rate
        sd.default.channels = 1
        sd.default.dtype = 'int16'
        sd.default.device = None
        sd.default.latency = 'low'

    if args.text is not None:
        rtf = []
        warmup = 10
        for  i in range(args.iter):
            if args.infer_device == "cuda":
                torch.cuda.synchronize()
            wav, _, _, _, rtf_i = tts(lexicon, g2p, preprocess_config, model, is_onnx, args)
            if i > warmup:
                rtf.append(rtf_i)
            if args.infer_device == "cuda":
                torch.cuda.synchronize()
            
            if args.play:
                sd.play(wav)
                sd.wait()

        if len(rtf) > 0:
            mean_rtf = np.mean(rtf)
            # print with 2 decimal places
            print("Average RTF: {:.2f}".format(mean_rtf))  
    else:
        print("Nothing to synthesize. Please provide a text file with --text")
    