'''
EfficientSpeech Text to Speech (TTS) demo.

To use microphone input with GUI interface, run:
    python3 asr_demo.py 

Dependencies:
    pip3 install pysimplegui
    pip3 install sounddevice 
    pip install Cython
'''

import torch
import yaml
import time
import numpy as np
import sounddevice as sd
import PySimpleGUI as sg

from model import EfficientFSModule
from utils.tools import get_args, write_to_file
from synthesize import get_lexicon_and_g2p, text2phoneme
#from scipy.io import wavfile

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
        #phoneme = np.array([text2phoneme(lexicon, g2p, args.text, preprocess_config)])
        #print(phoneme)
        #phoneme = np.pad(phoneme, ((0, 0), (0, 64 - phoneme.shape[1])), mode='constant', constant_values=196)
        #print("Phoneme shape", phoneme.shape)
        #ort_inputs = {ort_session.get_inputs()[0].name: phoneme}
        
        #wavs = ort_session.run(None, ort_inputs)[0]
        #print("wav shape", wavs.shape)
        #write_to_file(wavs, preprocess_config, args.wav_path)

        #wavs = (
        #    wavs * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        #    ).astype("int16")
        #wavs = [wav for wav in wavs]
        #for wav in wavs:
        #    wavfile.write("output.wav", sampling_rate, wav)

        is_onnx = True
    else:
        #pl_module = EfficientFSModule(preprocess_config=preprocess_config,)
        #ort_inputs = {input_name: np.random.randn(1, 64)}
        #ort_outs = ort_session.run(None, ort_inputs)

    #args.wav_path = None
        pl_module = EfficientFSModule(preprocess_config=preprocess_config, lr=args.lr,
                                      warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                      depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                      reduction=args.reduction, head=args.head,
                                      embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                      decoder_kernel_size=args.decoder_kernel_size,
                                      expansion=args.expansion, wav_path=args.out_folder,
                                      infer_device=args.infer_device)

    #phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config, lexicon=lexicon, g2p=g2p)
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
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
            start_time = time.time()
            phoneme = np.array([text2phoneme(lexicon, g2p, args.text, preprocess_config)])
            if is_onnx:
                if phoneme.shape[1] < 64:
                    phoneme = np.pad(phoneme, ((0, 0), (0, 64 - phoneme.shape[1])), mode='constant', constant_values=196)
                ort_inputs = {ort_session.get_inputs()[0].name: phoneme}
                wavs = ort_session.run(None, ort_inputs)[0]
            else:
                with torch.no_grad():
                    phoneme = torch.from_numpy(phoneme).long()  
                    wavs = pl_module({"phoneme": phoneme})
                    wavs = wavs.cpu().numpy()

            elapsed_time = time.time() - start_time
            wav = np.reshape(wavs, (-1, 1))
            message = f"Synthesis time: {elapsed_time:.2f} sec"
            wav_len = wav.shape[0] / sampling_rate
            message += f"\nVoice length: {wav_len:.2f} sec"
            real_time_factor = wav_len / elapsed_time
            message += f"\nReal time factor: {real_time_factor:.2f}"
            g_window['-TIME-'].update(message)
            g_window.refresh()

            sd.play(wav)
            sd.wait()
            write_to_file(wavs, preprocess_config, args.wav_path)
            #wavfile.write("output.wav", sampling_rate, wav)

        elif event == '-CLEAR-':
            multiline.update('')

    g_window.close()
