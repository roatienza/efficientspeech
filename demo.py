'''
EfficientSpeech Text to Speech (TTS) demo.

To use speaker with GUI interface, run:
    (Tagalog)
    ONNX: 
    (English LJ - Small)
      python3 demo.py --checkpoint checkpoints/small_v2_eng_attn.onnx --accelerator cpu \
        --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3

    (Tagalog ISIP - Tiny)
      python3 demo.py --checkpoint checkpoints/tiny_v2_tag_attn.onnx --preprocess-config \
        config/isip-preprocess.yaml
    
    Torch model:
    (English LJ - Small)
      python3 demo.py --checkpoint checkpoints/small_v2_eng_attn.ckpt --accelerator cpu \
        --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3

    (Tagalog ISIP - Tiny)
      python3 demo.py --checkpoint checkpoints/small_v2_tag_attn.ckpt --accelerator cpu \
        --infer-device cpu --head 1 --reduction 2 --expansion 1 --kernel-size 3 \
        --preprocess-config config/isip-preprocess.yaml

Dependencies:
    pip3 install pysimplegui
    pip3 install sounddevice 
'''

import torch
import yaml
import time
import numpy as np


from model import EfficientFSModule
from utils.tools import get_args, write_to_file
from synthesize import get_lexicon_and_g2p, text2phoneme

#from scipy.io import wavfile

def tts(lexicon, g2p, preprocess_config, pl_module, is_onnx, args, verbose=False):
    text = args.text.strip()
    text = text.replace('-', ' ')
    if text[-1] == ".":
        text = text[:-1]
    text += ". "
    phoneme = np.array(
        [text2phoneme(lexicon, g2p, text, preprocess_config, verbose=verbose)], dtype=np.int32)
    start_time = time.time()
    if is_onnx:
        # onnx is 3.5x faster than pytorch models
        phoneme_len = phoneme.shape[1]
        n_append = args.onnx_insize // phoneme_len
        phoneme = [phoneme] * (n_append + 1)
        phoneme = np.concatenate(phoneme, axis=1)
        phoneme = phoneme[:, :args.onnx_insize]

        ort_inputs = {ort_session.get_inputs()[0].name: phoneme}
        outputs = ort_session.run(None, ort_inputs)
        wavs = outputs[0]
        hop_len = preprocess_config["preprocessing"]["stft"]["hop_length"]
        duration = outputs[2]
        orig_duration = int(np.sum(np.round(duration.squeeze())[:phoneme_len])) * hop_len
        wavs = wavs[:, :orig_duration]
        duration = [orig_duration]
    else:
        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int()
            wavs, lengths = pl_module({"phoneme": phoneme})
            wavs = wavs.cpu().numpy()
            lengths = lengths.cpu().numpy()
        
    elapsed_time = time.time() - start_time
    if is_onnx:
        elapsed_time *= (wav.shape[0] / outputs[0].shape[1])
    wav = np.reshape(wavs, (-1, 1))

    message = f"Synthesis time: {elapsed_time:.2f} sec"
    wav_len = wav.shape[0] / sampling_rate
    message += f"\nVoice length: {wav_len:.2f} sec"
    real_time_factor = wav_len / elapsed_time
    message += f"\nReal time factor: {real_time_factor:.2f}"
    write_to_file(wavs, preprocess_config, lengths=lengths, \
        wav_path=args.wav_path, filename=args.wav_filename)
    
    if verbose:
        print(message)
    return wav, message, phoneme

if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
 
    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    is_onnx = False

    if "onnx" in args.checkpoint:
        #pl_module.load_from_onnx(args.checkpoint)
        import onnxruntime
        import onnx

        onnx_model = onnx.load(args.checkpoint)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(args.checkpoint)
        is_onnx = True
    else:
        pl_module = EfficientFSModule(preprocess_config=preprocess_config, infer_device=args.infer_device)

        pl_module = pl_module.load_from_checkpoint(args.checkpoint, 
                                                   preprocess_config=preprocess_config,
                                                   lr=args.lr, 
                                                   warmup_epochs=args.warmup_epochs, 
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
                                                   dropout=args.dropout,
                                                   verbose=args.verbose)
        print(pl_module.phoneme2mel.decoder)
        exit(0)
        pl_module.eval()
        if args.benchmark:
            if args.text is None:
                print("supply to convert to speech using --text")
                exit(1)
            from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
            # warmup
            for _ in range(10):
                _ = tts(lexicon, g2p, preprocess_config, pl_module, is_onnx, args)
            _, _, phoneme = tts(lexicon, g2p, preprocess_config, pl_module, is_onnx, args, verbose=True)
            #phoneme = np.array([text2phoneme(lexicon, g2p, args.text, preprocess_config)], dtype=np.int32)
            with torch.no_grad():
                phoneme = torch.from_numpy(phoneme).int().to(args.infer_device)
                #wavs, duration, _ = pl_module({"phoneme": phoneme})
                pl_module = pl_module.to(args.infer_device)
                flops = FlopCountAnalysis(pl_module, {"phoneme": phoneme})
                param = parameter_count(pl_module)
            print("FLOPS: {:,}".format(flops.total()))
            print("Parameters: {:,}".format(param[""]))
            print(flop_count_table(flops))
            exit(0)

    if args.text is not None:
        tts(lexicon, g2p, preprocess_config, pl_module, is_onnx, args)
        exit(0)


    import sounddevice as sd
    import PySimpleGUI as sg
    SIZE_X = 320
    SIZE_Y = 120

    sg.theme('DarkGrey13')
    graph = sg.Graph(canvas_size=(SIZE_X*2, SIZE_Y*2),
                     graph_bottom_left=(-(SIZE_X+5), -(SIZE_Y+5)),
                     graph_top_right=(SIZE_X+5, SIZE_Y+5),
                     background_color='black',
                     expand_x=False,
                     key='-GRAPH-')
    multiline = sg.Multiline(#size=(100,4),
                             expand_y=True,
                             expand_x=True,
                             background_color='black',
                             write_only=False,
                             pad=(10, 10),
                             no_scrollbar=True,
                             justification='left',
                             autoscroll=True,
                             font=("Helvetica", 36),
                             key='-OUTPUT-',)
    time_text = sg.Text("Voice", pad=(20, 20), font=("Helvetica", 20), key='-TIME-')
    play_button = sg.Button('Play', key='-PLAY-', font=("Helvetica", 20))
    clear_button = sg.Button('Clear', key='-CLEAR-', font=("Helvetica", 20))
    quit_button = sg.Button('Quit', key='-QUIT-', font=("Helvetica", 20))
    layout = [ [multiline], [graph, time_text], [play_button, clear_button, quit_button] ]
    #layout = [[sg.Sizer(0,500), sg.Column([[sg.Sizer(500,0)]] + layout, element_justification='c', pad=(0,0))]]
    g_window = sg.Window('Voice', layout, location=(0, 0), default_button_element_size=(2,1),
                         resizable=True).Finalize()
    g_window.Maximize()
    g_window.BringToFront()
    g_window.Refresh()
    
    sd.default.reset()
    sd.default.samplerate = sampling_rate
    sd.default.channels = 1
    sd.default.dtype = 'int16'
    sd.default.device = None
    sd.default.latency = 'low'

    while True:
        event, values = g_window.read()
        if event == sg.WIN_CLOSED or event == '-QUIT-':
            break
        elif event == '-PLAY-':
            current_frame = 0
            args.text = multiline.get()
            wav, message, _ = tts(lexicon, g2p, preprocess_config, pl_module, is_onnx, args)

            g_window['-TIME-'].update(message)
            g_window.refresh()

            sd.play(wav)
            sd.wait()

        elif event == '-CLEAR-':
            multiline.update('')

    g_window.close()
