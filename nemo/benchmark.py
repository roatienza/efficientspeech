
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
import torch
import argparse

#https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tts/intro.html

class tts(torch.nn.Module):
    def __init__(self,model_name, device):
        super().__init__()
        self.model = SpectrogramGenerator.from_pretrained(model_name=model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device

    def forward(self, x):
        if "tacotron2" in self.model_name:
            token_embedding = self.model.text_embedding(x).transpose(1, 2)
            token_len = torch.tensor([len(i) for i in x]).to(self.device)
            encoder_embedding = self.model.encoder(token_embedding=token_embedding, token_len=token_len)
            spec_pred_dec, gate_pred, alignments, pred_length = self.model.decoder(
                memory=encoder_embedding, memory_lengths=token_len
            )
            spec_pred_postnet = self.model.postnet(mel_spec=spec_pred_dec)
            return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
        
        return self.model.forward_for_export(x)


def get_args():
    parser = argparse.ArgumentParser()
    choices = ["cuda", "cpu",]
    parser.add_argument('--device', type=str, default=choices[0], \
                        choices=choices, help='device for inference')
    choices = ["tts_en_lj_mixertts", "tts_en_tacotron2",]
    parser.add_argument('--tts', type=str, default=choices[0], \
                        choices=choices, help='which tts model to use')
    # text
    parser.add_argument('--text', type=str, default="the quick brown fox jumps over the lazy dog", \
                        help='text to be synthesized')  
    # list all models
    parser.add_argument('--list-models', action='store_true', help='list all available models')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.list_models:
        print(SpectrogramGenerator.list_available_models())
        exit(0)

    tts_model = tts(model_name=args.tts, device=args.device).eval()
    parsed = tts_model.model.parse(args.text)

    print(type(tts_model))
    print(parsed)
    with torch.no_grad():
        flops = FlopCountAnalysis(tts_model, parsed)
        param = parameter_count(tts_model)
        print(flop_count_table(flops))
        print("FLOPS: {:,}".format(flops.total()))
        print("Parameters: {:,}".format(param[""]))