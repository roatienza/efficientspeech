
import yaml
import json
import os
from datamodule import LJSpeechDataModule
from utils.tools import get_args
from model import EfficientFSModule

from pytorch_lightning import Trainer

if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    datamodule.setup()

    #train_dataloader = datamodule.train_dataloader()

    #for i, (x, y) in enumerate(train_dataloader):
    #    print(x["phoneme"].shape)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        pitch_stats = stats["pitch"][:2]
        energy_stats = stats["energy"][:2]
        print("Pitch min/max", pitch_stats)
        print("Energy min/max", energy_stats)

    phoneme2mel = EfficientFSModule(preprocess_config=preprocess_config, lr=args.lr, 
                                    warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                    depth=args.depth, reduction=args.reduction, head=args.head,
                                    embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                    expansion=args.expansion)

    trainer = Trainer(accelerator=args.accelerator, devices=args.devices, 
                     precision=args.precision,
                     strategy="ddp", max_epochs=args.max_epochs,)

    trainer.fit(phoneme2mel, datamodule=datamodule)
    #trainer.test(phoneme2mel, datamodule=datamodule)