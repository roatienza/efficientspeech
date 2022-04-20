
import yaml
from datamodule import LJSpeechDataModule
from utils.tools import get_args

if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    for i, (x, y) in enumerate(train_dataloader):
        print(x["text"])
        print(x["phoneme"])
        print(x["pitch"])
        print(x["energy"])
        print(x["duration"])
        print(y["mel"])
        print(y["mel_len"])
        print("\n")
        break