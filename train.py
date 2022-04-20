
import os
import torch
import yaml
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_config", type=str, default="config/preprocess.yaml")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    return args


def get_dataloader(dataset, batch_size):
    num_workers = torch.cuda.device_count() * 4
    if "cuda" in str(device):
        print("num_workers: ", num_workers)
        kwargs = {'num_workers': num_workers, 'pin_memory': True}
    else:
        kwargs = {}
    return DataLoader(dataset,
                      shuffle=True,
                      batch_size=batch_size,
                      collate_fn=dataset.collate_fn,
                      **kwargs)

def main(args):
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    #train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    # Get dataset
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    loader = get_dataloader(dataset, args.batch_size)

    dataset = Dataset("val.txt", preprocess_config, train_config, sort=True, drop_last=True)
    val_loader = get_dataloader(dataset, args.batch_size)

    torch.backends.cudnn.benchmark = True
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        pitch_stats = stats["pitch"][:2]
        energy_stats = stats["energy"][:2]
        print("Pitch min/max", pitch_stats)
        print("Energy min/max", energy_stats)

    phoneme_encoder = PhonemeEncoder(pitch_stats=pitch_stats,
                                     energy_stats=energy_stats,
                                     depth=args.depth,
                                     reduction=args.reduction,
                                     head=args.head,
                                     embed_dim=args.embed_dim,
                                     kernel_size=args.kernel_size,
                                     expansion=args.expansion)

    mel_decoder = MelDecoder(dims=phoneme_encoder.dims,
                             kernel_size=args.kernel_size)

    phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                              decoder=mel_decoder,
                              distill=args.distill).to(device)


    if args.distill:
        assert args.checkpoint is not None
        print("Loading model checkpoint ..." , args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        phoneme2mel.load_state_dict(checkpoint["phoneme2mel"])
        phoneme2mel.eval()
        wav_decoder = WavDecoder(dims=phoneme_encoder.dims,
                                 kernel_size=args.kernel_size).to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = ScheduledOptim(wav_decoder, 
                                   train_config, 
                                   model_config, 
                                   args.restore_step)
    else:
        phoneme2mel.train()
        optimizer = ScheduledOptim(phoneme2mel, 
                                   train_config, 
                                   model_config, 
                                   args.restore_step)

    log  = get_parameter_count(phoneme_encoder, "Phoneme Encoder")
    log += get_parameter_count(mel_decoder, "Mel Decoder")
    log += get_parameter_count(phoneme2mel, "Phoneme to Mel")
    if args.distill:
        log += get_parameter_count(wav_decoder, "Wav Decoder")

    #model = torch.nn.DataParallel(model)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    log += get_parameter_count(vocoder, "HifiGAN Vocoder")

    train_log_path, gen_wav_path = get_paths(train_config, args)

    log_message(log, train_log_path, args)
    # Training
    step = args.restore_step + 1
    epoch = 1
    
    #grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    #grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]

    total_step = train_config["step"]["total_step"]
    log_step   = train_config["step"]["log_step"]
    save_step  = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    #val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    for batchs in val_loader:
        for batch in batchs:
            val_batch = to_device(batch, device)
            break
        break

    loss_m = AverageMeter()
    if not args.distill:
        mel_loss_m = AverageMeter()
        pitch_loss_m = AverageMeter()
        energy_loss_m = AverageMeter()
        duration_loss_m = AverageMeter()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                # Forward
                #ids = batch[0]
                phoneme = batch[3]
                phoneme_len = batch[4]
                max_phoneme_len = batch[5]
                mel_len = batch[7]
                max_mel_len = batch[8]
                pitch_target = batch[9]
                energy_target = batch[10]
                duration_target = batch[11]

                pred = phoneme2mel(phoneme, 
                                   phoneme_len=phoneme_len, 
                                   max_phoneme_len=max_phoneme_len, 
                                   pitch_target=pitch_target,
                                   energy_target=energy_target,
                                   duration_target=duration_target, 
                                   mel_len=mel_len,
                                   max_mel_len=max_mel_len)

                    #pitch_pred, energy_pred, duration_pred, len_pred, features, phoneme_mask, mask = pred
                    #pred += phoneme2mel(features, mask, mel_len=mel_len, max_mel_len=max_mel_len)


                #for p in model.parameters(): p.grad = None
                optimizer.zero_grad()
                
                if args.distill:
                    mel_pred, len_pred, features = pred
                    print(features.size())
                    wav_pred = wav_decoder(features)
                    print(wav_pred.size())
                    exit(0)
                else:
                    mel_loss, pitch_loss, energy_loss, duration_loss = phoneme2mel_losses(gt=batch, pred=pred)
                    loss = (args.mel_loss_weight * mel_loss) + \
                           (args.pitch_loss_weight * pitch_loss) +  \
                           (args.energy_loss_weight * energy_loss) + \
                           duration_loss
                    mel_loss_m.update(mel_loss.item())
                    pitch_loss_m.update(pitch_loss.item())
                    energy_loss_m.update(energy_loss.item())
                    duration_loss_m.update(duration_loss.item())

                
                loss.backward()
                loss_m.update(loss.item())

                # Clipping gradients to avoid gradient explosion
                #nn.utils.clip_grad_norm_(model.parameters(), 1.)

                # Update weights
                lr = optimizer.step_and_update_lr()

                if step % log_step == 0:
                    if args.distill:
                        distll_loss_m.reset()
                        message = ""
                    else:
                        losses = (loss_m, mel_loss_m, pitch_loss_m, energy_loss_m, duration_loss_m)
                        losses = [l.avg for l in losses]
                        losses.append(lr)
                        message = phoneme2mel_logs(losses, step, total_step, train_log_path, args)

                        loss_m.reset()
                        mel_loss_m.reset()
                        pitch_loss_m.reset()
                        energy_loss_m.reset()
                        duration_loss_m.reset()

                    outer_bar.write(message)

                if step % save_step == 0:
                    if args.distill:
                        pass
                    else:
                        torch.save({"phoneme_encoder": phoneme_encoder.state_dict(),
                                    "mel_decoder": mel_decoder.state_dict(),
                                    "phoneme2mel": phoneme2mel.state_dict(),
                                    "optimizer": optimizer._optimizer.state_dict(),},
                                    os.path.join(train_log_path, "{}.pth.tar".format(step),),)
                if step % synth_step == 0:
                    if args.distill:
                        pass
                    else:
                        phoneme2mel_wavs(val_batch,
                                         phoneme2mel, 
                                         vocoder, 
                                         model_config, 
                                         preprocess_config, 
                                         gen_wav_path, 
                                         args)

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


def phoneme2mel_wavs(val_batch,
                     phoneme2mel, 
                     vocoder, 
                     model_config, 
                     preprocess_config, 
                     gen_wav_path, 
                     args):
    phoneme = val_batch[3]
    phoneme_len = val_batch[4]
    max_phoneme_len = val_batch[5]
    mel = val_batch[6]
    mel_len = val_batch[7]
    max_mel_len = val_batch[8]

    phoneme2mel.eval()
    with torch.no_grad():
        mel_pred, len_pred = phoneme2mel(phoneme, 
                                         phoneme_len=phoneme_len, 
                                         max_phoneme_len=max_phoneme_len)
    #_, _, _, len_pred, _, _, _, mel_pred, _ = pred
    phoneme2mel.train()

    synth_test_samples(mel=mel,
                       mel_len=mel_len,
                       mel_pred=mel_pred,
                       mel_len_pred=len_pred,
                       vocoder=vocoder,
                       model_config=model_config,
                       preprocess_config=preprocess_config,
                       wav_path=gen_wav_path,
                       count=args.count
                       )


def phoneme2mel_logs(losses, step, total_step, train_log_path, args):
    message1 = "Step {}/{}, ".format(step, total_step)
    message2 = "Loss: {:.3f}, Mel: {:.3f}, Pitch: {:.3f}, Energy: {:.3f}, Duration: {:.3f}, LR: {:.3e}".format(*losses)
    message3 = "| Depth: " + str(args.depth) + ", Reduce: " + str(args.reduction) + ", Head: " + str(args.head) + ", Embed: " + str(args.embed_dim) 
    message3 += ", Kern: " + str(args.kernel_size) + ", Exp: " + str(args.expansion)  + ", Dist: " + str(args.distill) 
    message = message1 + message2 + message3

    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
        f.write(message + "\n")

    return message


def phoneme2mel_losses(gt, pred):
    mel = gt[6]
    pitch = gt[9]
    energy = gt[10]
    duration = gt[11]

    pitch_pred, energy_pred, duration_pred, _, _, phoneme_mask, _, mel_pred, mel_mask = pred
    #mel_pred, pitch_pred, energy_pred, duration_pred, _, phoneme_mask, mel_mask, _ = pred
    mel_mask = ~mel_mask
    mel_mask = mel_mask.unsqueeze(-1)
    target = mel.masked_select(mel_mask)
    pred = mel_pred.masked_select(mel_mask)
    mel_loss = nn.L1Loss()(pred, target)
    
    phoneme_mask = ~phoneme_mask

    pitch_pred = pitch_pred[:,:pitch.shape[-1]]
    pitch_pred = torch.squeeze(pitch_pred)
    #pitch      = torch.tanh(pitch)
    pitch      = pitch.masked_select(phoneme_mask)
    pitch_pred = pitch_pred.masked_select(phoneme_mask)
    pitch_loss = nn.MSELoss()(pitch_pred, pitch)

    energy_pred = energy_pred[:,:energy.shape[-1]]
    energy_pred = torch.squeeze(energy_pred)
    #energy      = torch.tanh(energy)
    energy      = energy.masked_select(phoneme_mask)
    energy_pred = energy_pred.masked_select(phoneme_mask)
    energy_loss = nn.MSELoss()(energy_pred, energy)

    duration_pred = duration_pred[:,:duration.shape[-1]]
    duration_pred = torch.squeeze(duration_pred)
    duration      = duration.masked_select(phoneme_mask)
    duration_pred = duration_pred.masked_select(phoneme_mask)
    duration      = torch.log(duration.float() + 1)
    duration_pred = torch.log(duration_pred.float() + 1)
    duration_loss = nn.MSELoss()(duration_pred, duration)

    return mel_loss, pitch_loss, energy_loss, duration_loss


if __name__ == "__main__":
    args = get_args()
    main(args)
