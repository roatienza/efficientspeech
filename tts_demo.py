'''
Text to Speech (TTS) demo.

To use microphone input with GUI interface, run:
    python3 asr_demo.py 

Dependencies:
    pip3 install pysimplegui
    pip3 install sounddevice 
    pip install Cython
    pip install nemo_toolkit['all']
'''

from tkinter import font
import torch
#import nemo.collections.asr as nemo_asr
import yaml
#import gc

import time

import numpy as np
import threading

from queue import Queue
#from decoder import ChunkBufferDecoder
#from utils import AudioChunkIterator
from model import EfficientFSModule

from utils.tools import get_args
from synthesize import load_module, synthesize, get_lexicon_and_g2p
from scipy.io import wavfile

def audio_callback(outdata, frames, time, status):
    #global g_chunk
    #global g_num_samples
    # size is (30,1). Samples every 10 only.
    # fill the graph queue with data
    global current_frame
    global g_draw_q
    
    if status:
        print(status)
    chunksize = min(len(wav) - current_frame, frames)
    outdata[:chunksize] = wav[current_frame:current_frame + chunksize]
    g_draw_q.put(outdata[::10, 0])
    if chunksize < frames:
        outdata[chunksize:] = 0
        raise sd.CallbackStop()
    current_frame += chunksize


class GraphThread(threading.Thread):
    def __init__(self, size_x=320, size_y=240):
        super(GraphThread, self).__init__()
        self.waveform = np.zeros((size_x,), dtype='float32')
        self.x_off = -size_x
        self.y_off = size_y
    
    def run(self):
        global g_run_thread
        global g_draw_q
        global g_window

        total = 0
        while g_run_thread:
            data = g_draw_q.get()
            shift = data.shape[0]
            self.waveform = np.roll(self.waveform, -shift, axis=0)
            self.waveform[-shift:] = data
            total += shift
            if total < self.waveform.shape[0]:
                continue
        
            g_window['-GRAPH-'].Erase()
            x = 0
            total = 0

            for y in self.waveform:
                if not g_run_thread:
                    return
                y = int(y * 500)
                if x > 0 and g_run_thread:
                    g_window['-GRAPH-'].draw_line((self.x_off+prev_x, prev_y), (self.x_off+x, y), color='white')
                prev_x = x
                prev_y = y
                x += 2


class TTSThread(threading.Thread):
    def __init__(self, mdeol):
        super(TTSThread, self).__init__()
        self.device = device


    def run(self):
        global g_run_thread
        global g_audio_q
        global g_window

        buffer_len = self.sample_rate * self.buffer_len_in_secs
        sampbuffer = np.zeros([buffer_len], dtype=np.float32)
        total_runtime = 0
        n_loops = 0
        old_transcription = ""
        while g_run_thread:
            waveform = g_audio_q.get()
            if g_run_thread is False:
                return
            

            start_time = time.time()


            elapsed_time = time.time() - start_time
            total_runtime += elapsed_time
            n_loops += 1
            ave_pred_time = total_runtime / n_loops

            if g_run_thread is False:
                return

            
            #g_window['-OUTPUT-'].update(transcription)
            g_window['-TIME-'].update(f"{ave_pred_time:.2f} sec")
            g_window.refresh()
            
        
# main routine
if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    args.wav_path = None
    pl_module = EfficientFSModule(preprocess_config=preprocess_config, lr=args.lr,
                                  warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                  depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                  reduction=args.reduction, head=args.head,
                                  embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                  decoder_kernel_size=args.decoder_kernel_size,
                                  expansion=args.expansion, wav_path=args.out_folder,
                                  infer_device=args.infer_device)
    if "onnx" in args.checkpoint:
        #pl_module.load_from_onnx(args.checkpoint)

        import onnxruntime

        ort_session = onnxruntime.InferenceSession(args.checkpoint)
        input_name = ort_session.get_inputs()[0].name
        #phoneme = torch.randint(low=1, high=10, size=(1,256)).long()
        phoneme = np.random.randint(low=1, high=10, size=(1, 256))
        # random tensor of type bool
        #phoneme_mask = torch.zeros(1, 256)
        x = {"phoneme": phoneme, }
        ort_inputs = {input_name: x}
        
        ort_outs = ort_session.run(None, ort_inputs)
        #print(ort_outs.keys())
        exit(0)
        #ort_inputs = {input_name: np.random.randn(1, 64)}
        #ort_outs = ort_session.run(None, ort_inputs)

    phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config)
    if args.onnx or args.jit:
        exit(0)
    else:
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
    
    #g_audio_q = Queue()
    g_draw_q = Queue()

    g_run_thread = True
    g_chunk = None
    #g_num_samples = args.chunk_len_in_secs * args.sample_rate

    graph_thread = GraphThread(SIZE_X, SIZE_Y)
    #graph_thread.start()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = get_model(device)
    #stride = args.stride # 8 for Citrinet
    #asr_thread = ASRThread(device, model, args.sample_rate,
    #                       args.chunk_len_in_secs, args.context_len_in_secs, 
    #                       stride=stride)
    #asr_thread.start()
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    
    sd.default.reset()
    sd.default.samplerate = sampling_rate
    sd.default.channels = 1
    sd.default.dtype = 'int16'
    sd.default.device = None
    sd.default.latency = 'low'
    
    
    #phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config)
    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)

    while True:
        event, values = g_window.read()
        if event == sg.WIN_CLOSED or event == '-QUIT-':
            g_run_thread = False
            #graph_thread.join(1)
            #asr_thread.join(3)
            break
        elif event == '-PLAY-':
            #tts_event = threading.Event()
            current_frame = 0
            args.text = multiline.get()
            start_time = time.time()
            wav = synthesize(lexicon, g2p, args, phoneme2mel, hifigan,
                             preprocess_config=preprocess_config)

            elapsed_time = time.time() - start_time

            #g_window['-OUTPUT-'].update(transcription)

            wav = np.reshape(wav, (-1, 1))
            message = f"Synthesis time: {elapsed_time:.2f} sec"
            wav_len = wav.shape[0] / sampling_rate
            message += f"\nVoice length: {wav_len:.2f} sec"
            real_time_factor = wav_len / elapsed_time
            message += f"\nReal time factor: {real_time_factor:.2f}"
            g_window['-TIME-'].update(message)
            g_window.refresh()

            sd.play(wav)
            sd.wait()
            wavfile.write("output.wav", sampling_rate, wav)
            #stream = sd.OutputStream(samplerate=sampling_rate,
            #                         channels=1,
            #                         callback=audio_callback)
            #print("wav shape", wav.shape)
            
            #with stream:
            #    tts_event.wait()
                #stream.write(wav)
                #sd.play(wav)


        elif event == '-CLEAR-':
            multiline.update('')

    #stream.close()
    g_window.close()
