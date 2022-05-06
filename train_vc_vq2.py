import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
##
import soundfile as sf
##

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  VQVoiceConverter2,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '6000'

  hps = utils.get_hparams()
#  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
##
  run(0, 1, hps)
##

def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

##  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
##
# for windows
  dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
##
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files_notext, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
##  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
##      collate_fn=collate_fn)
##
  train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler, prefetch_factor=hps.train.batch_size)
##
  index_to_sid = train_dataset.get_unique_sids().cuda(rank)
  sid_to_index = torch.zeros(hps.data.n_speakers, dtype=torch.int64, device=index_to_sid.device)
  sid_to_index[index_to_sid] = torch.arange(index_to_sid.size(0), device=index_to_sid.device)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files_notext, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  net_g = VQVoiceConverter2(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  if hps.fine_flag:
    logger.info('Load model : '+str(hps.fine_model_g))
    logger.info('Load model : '+str(hps.fine_model_d))
    _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_g, net_g)
    _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_d, net_d)
    epoch_str = 1
    global_step = 0
  else:
    try:
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
      global_step = (epoch_str - 1) * len(train_loader)
    except:
      epoch_str = 1
      global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], scaler, [train_loader, eval_loader],
                         sid_to_index, index_to_sid, logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], scaler, [train_loader, None],
                         sid_to_index, index_to_sid, None, None)
    scheduler_g.step()
    scheduler_d.step()
  utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir,
                        "G_{}.pth".format(global_step)))
  utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir,
                        "D_{}.pth".format(global_step)))


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, sid_to_index, index_to_sid, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (_, _, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader, desc="Epoch {}".format(epoch))):
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, ids_slice, z_mask, latent_loss = net_g(spec, spec_lengths, speakers)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        # TODO: rename to "latent_loss"
        loss_kl = 0.25 * latent_loss.mean() 
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen_all, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, sid_to_index, index_to_sid, writer_eval, logger)
##
      if global_step % hps.train.save_interval == 0:
##
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, sid_to_index, index_to_sid, writer_eval, logger):
    generator.eval()
    scalar_dict = {}
    scalar_dict.update({"loss/g/mel": 0.0, "loss/g/dur": 0.0, "loss/g/kl": 0.0})
    with torch.no_grad():
      #evalのデータセットを一周する
      for batch_idx, (_, _, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(eval_loader, desc="Epoch {}".format("eval"))):
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)
        #autocastはfp16のおまじない
        with autocast(enabled=hps.train.fp16_run):
          #Generator
          y_hat, ids_slice, z_mask, latent_loss = generator(spec, spec_lengths, speakers)

          mel = spec_to_mel_torch(
              spec, 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate,
              hps.data.mel_fmin, 
              hps.data.mel_fmax)
          y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
          y_hat_mel = mel_spectrogram_torch(
              y_hat.squeeze(1), 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
          )
          batch_num = batch_idx

          y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

        with autocast(enabled=hps.train.fp16_run):
          with autocast(enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            # TODO: rename to "latent_loss"
            loss_kl = 0.25 * latent_loss.mean() 

        scalar_dict["loss/g/mel"] = scalar_dict["loss/g/mel"] + loss_mel
        scalar_dict["loss/g/kl"] = scalar_dict["loss/g/kl"] + loss_kl
      
      #lossをepoch1周の結果をiter単位の平均値に
      scalar_dict["loss/g/mel"] = scalar_dict["loss/g/mel"] / (batch_num+1)
      scalar_dict["loss/g/kl"] = scalar_dict["loss/g/kl"] / (batch_num+1)
      logger.info("loss/g/mel : {} loss/g/dur : {} loss/g/kl : {}".format(str(scalar_dict["loss/g/mel"]), str(scalar_dict["loss/g/dur"]), str(scalar_dict["loss/g/kl"])))

      #evalデータセットの先頭を取得
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)

        # remove else
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break

      y_hat, mask, *_ = generator.module.infer(spec, spec_lengths, speakers, speakers)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )

      # Convert voice to another speaker
      another_speakers = index_to_sid[(sid_to_index[speakers] + 1) % index_to_sid.size(0)]
      vc_y_hat, vc_mask, *_ = generator.module.infer(spec, spec_lengths, speakers, another_speakers)
      vc_y_hat_lengths = vc_mask.sum([1,2]).long() * hps.data.hop_length

      vc_y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )

    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
      "convert/mel": utils.plot_spectrogram_to_numpy(vc_y_hat_mel[0].cpu().numpy()),
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]],
      "convert/audio": vc_y_hat[0,:,:vc_y_hat_lengths[0]],
    }
    ##
    sf.write(os.path.join(hps.model_dir, 'voice_vc_{0:06d}.wav'.format(global_step)), vc_y_hat[0,:,:vc_y_hat_lengths[0]].transpose(0, 1).detach().cpu().numpy(), hps.data.sampling_rate)
    ##
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      scalars=scalar_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
  main()
