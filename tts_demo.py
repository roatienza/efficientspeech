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

import torch
#import nemo.collections.asr as nemo_asr

import gc
#import sounddevice as sd
import time
import argparse
import numpy as np
import threading
#import PySimpleGUI as sg
from queue import Queue
from decoder import ChunkBufferDecoder
from utils import AudioChunkIterator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--chunk_len_in_secs", default=4, type=int)
    parser.add_argument("--context_len_in_secs", default=2, type=int)
    parser.add_argument("--stride", default=4, type=int)

    parser.add_argument("--rpi", default=False, action="store_true")
    args = parser.parse_args()
    return args

def get_model(device):
    torch.cuda.empty_cache()
    gc.collect()
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large", map_location=device)
    model = model.to(device)
    return model

def audio_callback(indata, frames, time, status):
    global g_chunk
    global g_num_samples
    # size is (30,1). Samples every 10 only.
    # fill the graph queue with data
    g_draw_q.put(indata[::10, 0])
    if g_chunk is None:
        g_chunk = indata.copy()
    else:
        g_chunk = np.concatenate((g_chunk, indata), axis=0)
        # if one chunk, enqueue it 
        if g_chunk.shape[0] > g_num_samples:
            g_chunk = g_chunk.squeeze()
            g_audio_q.put(g_chunk[:g_num_samples])
            g_chunk = None


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


class ASRThread(threading.Thread):
    def __init__(self, device, model, sample_rate, chunk_len_in_secs, context_len_in_secs, stride):
        super(ASRThread, self).__init__()
        self.device = device
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_len_in_secs = chunk_len_in_secs
        self.context_len_in_secs = context_len_in_secs
        self.buffer_len_in_secs = chunk_len_in_secs + 2 * context_len_in_secs
        self.stride = stride
        self.decoder = ChunkBufferDecoder(asr_model=self.model,
                                          chunk_len_in_secs=self.chunk_len_in_secs,
                                          buffer_len_in_secs=self.buffer_len_in_secs,
                                          stride=self.stride)

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
            chunk_reader = AudioChunkIterator(waveform, self.chunk_len_in_secs, self.sample_rate)
            chunk_len = self.sample_rate * self.chunk_len_in_secs
            buffer_list = []
            for chunk_ in chunk_reader:
                sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
                sampbuffer[-chunk_len:] = chunk_
                buffer_list.append(np.array(sampbuffer))

            start_time = time.time()
            transcription = self.decoder.transcribe_buffers(buffer_list, plot=False)
            transcription = transcription.strip()

            elapsed_time = time.time() - start_time
            total_runtime += elapsed_time
            n_loops += 1
            ave_pred_time = total_runtime / n_loops

            if g_run_thread is False:
                return

            if transcription == old_transcription:
                self.decoder.all_preds = []
                self.decoder.all_targets = []
                g_window['-OUTPUT-'].update("")
                g_window.refresh()
                continue
            else:
                old_transcription = transcription
            
            g_window['-OUTPUT-'].update(transcription)
            g_window['-TIME-'].update(f"{ave_pred_time:.2f} sec")
            g_window.refresh()
            
        
# main routine
if __name__ == "__main__":
    args = get_args()
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
                             write_only=True,
                             pad=(10, 10),
                             no_scrollbar=True,
                             justification='left',
                             autoscroll=True,
                             font=("Helvetica", 60),
                             key='-OUTPUT-',)
    time_text = sg.Text("Voice", pad=(20, 20), font=("Helvetica", 20), key='-TIME-')
    layout = [ [multiline], [graph], [time_text],]
    #layout = [[sg.Sizer(0,500), sg.Column([[sg.Sizer(500,0)]] + layout, element_justification='c', pad=(0,0))]]
    g_window = sg.Window('Voice', layout, location=(0, 0), 
                         resizable=True).Finalize()
    g_window.Maximize()
    g_window.BringToFront()
    g_window.Refresh()
    
    g_audio_q = Queue()
    g_draw_q = Queue()

    g_run_thread = True
    g_chunk = None
    g_num_samples = args.chunk_len_in_secs * args.sample_rate

    graph_thread = GraphThread(SIZE_X, SIZE_Y)
    graph_thread.start()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(device)
    stride = args.stride # 8 for Citrinet
    asr_thread = ASRThread(device, model, args.sample_rate,
                           args.chunk_len_in_secs, args.context_len_in_secs, 
                           stride=stride)
    asr_thread.start()

    sd.default.samplerate = args.sample_rate
    sd.default.channels = 1
    stream = sd.InputStream(device=None, callback=audio_callback)

    with stream:
        while True:
            event, values = g_window.read()
            if event == sg.WIN_CLOSED:
                g_run_thread = False
                graph_thread.join(1)
                asr_thread.join(3)
                break

    stream.close()
    g_window.close()
