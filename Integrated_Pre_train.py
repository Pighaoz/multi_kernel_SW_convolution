from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random
import numpy as np

# Add parent directory to sys.path to access util, networks, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from Pre_dataset import FaultDataset
from util import weakCustomTransform, strongCustomTransform
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.swmodel import ContMixDualStream, ContMixContrastive
from losses import SupConLoss, SupConGapLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for pre-training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--vis_freq', type=int, default=10,
                        help='visualization frequency (batches)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='custom',
                        choices=['custom'], help='dataset')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs', help='path to custom dataset directory')
    parser.add_argument('--seq_length', type=int, default=1024, help='sequence length for time series')
    parser.add_argument('--normalize', action='store_true', help='enable normalization in dataset')
    
    # Pre-training specific
    parser.add_argument('--max_samples_per_class', type=int, default=2000, help='samples per class for pre-training')

    # transformations parameters
    parser.add_argument('--scale_mu', type=float, default=2, help='mu for gaussian scaling in weak transform')
    parser.add_argument('--scale_sigma', type=float, default=1.1, help='sigma for gaussian scaling in weak transform')
    parser.add_argument('--jitter_loc', type=float, default=0.0, help='loc for jitter noise in weak transform')
    parser.add_argument('--jitter_sigma', type=float, default=0.8, help='sigma for jitter noise in weak transform')
    parser.add_argument('--trans_max_shift', type=int, default=3, help='max circular shift for translation in strong transform')
    parser.add_argument('--snr_db', type=float, default=6, help='SNR (dB) for additive noise in strong transform')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--gap_weight', type=float, default=0.0,
                        help='weight for gap term')

    opt = parser.parse_args()

    # Set paths locally within the current folder
    opt.model_path = './save/PreTrain_models'
    opt.tb_path = './save/PreTrain_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_bsz_{}_temp_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.learning_rate,
               opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.batch_size >= 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.exists(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    weak_transform = weakCustomTransform(
        mu=opt.scale_mu,
        delta_s=opt.scale_sigma,
        loc=opt.jitter_loc,
        jitter_sigma=opt.jitter_sigma
    )
    strong_transform = strongCustomTransform(
        max_shift=opt.trans_max_shift,
        snr_db=opt.snr_db
    )
    two_crop_transform = TwoCropTransform(weak_transform, strong_transform)

    train_dataset = FaultDataset(
        data_dir=opt.data_dir,
        seq_length=opt.seq_length,
        transform=two_crop_transform,
        mode='train',
        normalize=opt.normalize,
        max_samples_per_class=opt.max_samples_per_class,
        seed=42,
        subset_mode='first_half'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader if len(train_dataset) > 0 else None


def set_model(opt):
    encoder = ContMixDualStream(
        in_channels=3,
        embed_dim=128,
        num_heads=8,
        patch_size=8,
        depth=3,
        kernel_sizes=[7,15,31],
        num_classes=None
    )
    model = ContMixContrastive(encoder=encoder, proj_dim=64)

    if opt.gap_weight and opt.gap_weight > 0:
        criterion = SupConGapLoss(
            temperature=opt.temp,
            contrast_mode='all',
            base_temperature=opt.temp,
            gap_weight=opt.gap_weight,
            normalize=True
        )
    else:
        criterion = SupConLoss(
            temperature=opt.temp,
            contrast_mode='all',
            base_temperature=opt.temp
        )

    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, logger):
    """one epoch training with enhanced visualization"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # 新增监控指标
    pos_similarities = AverageMeter()
    neg_similarities = AverageMeter()
    similarity_gaps = AverageMeter()
    temperatures = AverageMeter()

    for idx, (views, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_step = (epoch - 1) * len(train_loader) + idx
        images = torch.cat([views[0], views[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # === 1. 获取模型输出 ===
        # 假设 model 返回 (proj, enc)
        output = model(images)
        
        if isinstance(output, tuple):
            proj_output, enc_output = output
        else:
            # 兼容旧模型只返回一个的情况
            proj_output = output
            enc_output = output

        # === 2. 数据格式处理 (Reshape to [B, 2, D]) ===
        # 处理 Projection 特征 (用于 SupCon Loss)
        f1_p, f2_p = torch.split(proj_output, [bsz, bsz], dim=0)
        proj_features = torch.cat([f1_p.unsqueeze(1), f2_p.unsqueeze(1)], dim=1) # (B, 2, D_proj)

        # 处理 Encoder 特征 (用于 Gap Loss)
        f1_e, f2_e = torch.split(enc_output, [bsz, bsz], dim=0)
        enc_features = torch.cat([f1_e.unsqueeze(1), f2_e.unsqueeze(1)], dim=1) # (B, 2, D_enc)

        # === 3. 计算 Loss ===
        # 根据 method 决定是否使用标签
        if opt.method == 'SupCon':
            # 有监督模式：传入 labels
            loss_labels = labels
        elif opt.method == 'SimCLR':
            # 无监督模式：不传 labels (设为 None)
            loss_labels = None
        else:
            raise ValueError(f"Unknown method: {opt.method}")

        if opt.gap_weight > 0:
            loss = criterion((proj_features, enc_features), loss_labels)
        else:
            loss = criterion(proj_features, loss_labels)

        # === 4. 更新指标 (直接从 Loss 对象中获取) ===
        losses.update(loss.item(), bsz)
        
        # 只有当 criterion 计算了 Gap (即使用了 SupConGapLoss) 时才更新监控指标
        if hasattr(criterion, 'last_gap'):
            pos_similarities.update(criterion.last_pos_sim, bsz)
            neg_similarities.update(criterion.last_neg_sim, bsz)
            similarity_gaps.update(criterion.last_gap, bsz)
            
            # 记录 Loss 分量
            current_gap = criterion.last_gap
            gap_term = opt.gap_weight * current_gap
            sup_part = loss.item() + gap_term
            logger.log_value('loss_components/sup_loss', sup_part, current_step)
            logger.log_value('loss_components/gap_term', gap_term, current_step)

        # 记录温度
        if hasattr(criterion, 'current_temperature'):
            current_temp = float(criterion.current_temperature)
            temperatures.update(current_temp, bsz)
        else:
            current_temp = float(opt.temp) # Fallback

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        


        # 1. 记录每个batch的损失
        logger.log_value('train/loss_iter', loss.item(), current_step)

        if idx % opt.print_freq == 0:
            # 仅在有值时记录
            if hasattr(criterion, 'last_gap'):
                logger.log_value('metrics/pos_similarity', criterion.last_pos_sim, current_step)
                logger.log_value('metrics/neg_similarity', criterion.last_neg_sim, current_step)
                logger.log_value('metrics/similarity_gap', criterion.last_gap, current_step)
            
            if hasattr(criterion, 'current_temperature'):
                logger.log_value('metrics/temperature', current_temp, current_step)
            
            logger.log_value('metrics/learning_rate', optimizer.param_groups[0]['lr'], current_step)
        


        # 2. 记录梯度信息（每隔一定批次）
        if idx % opt.vis_freq == 0:
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    # logger.log_value(f'gradients/{name}_norm', grad_norm, current_step) # 可选：减少日志量

            logger.log_value('gradients/total_norm', total_grad_norm, current_step)
            
        # print info
        if (idx + 1) % opt.print_freq == 0:
            temp_info = ""
            if hasattr(criterion, 'current_temperature'):
                temp_info = f'Temp {current_temp:.4f}\t'
                
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'PosSim {pos_sim.val:.3f} ({pos_sim.avg:.3f})\t'
                  'NegSim {neg_sim.val:.3f} ({neg_sim.avg:.3f})\t'
                  'Gap {gap.val:.3f} ({gap.avg:.3f})\t'
                  '{temp_info}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                pos_sim=pos_similarities, neg_sim=neg_similarities, 
                gap=similarity_gaps, temp_info=temp_info))
            sys.stdout.flush()
    # ========== 记录epoch级别的统计 ========== #
    logger.log_value('train/loss_epoch', losses.avg, epoch)
    logger.log_value('train/pos_similarity_epoch', pos_similarities.avg, epoch)
    logger.log_value('train/neg_similarity_epoch', neg_similarities.avg, epoch)
    logger.log_value('train/similarity_gap_epoch', similarity_gaps.avg, epoch)
    
    if hasattr(criterion, 'current_temperature'):
        logger.log_value('train/temperature_epoch', temperatures.avg, epoch)

    # 打印epoch总结
    print(f'Epoch {epoch} Summary: '
          f'Loss: {losses.avg:.4f}, '
          f'PosSim: {pos_similarities.avg:.3f}, '
          f'NegSim: {neg_similarities.avg:.3f}, '
          f'Gap: {similarity_gaps.avg:.3f}')
                
    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    best_loss = float('inf')  # 初始化为正无穷大
    best_epoch = 0

    # 使用TensorBoard记录器
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_value('model/total_params', total_params, 0)
    logger.log_value('model/trainable_params', trainable_params, 0)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt,logger)
        time2 = time.time()
        epoch_time = time2 - time1
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logger.log_value('train/epoch_time', epoch_time, epoch)
        logger.log_value('train/best_loss', best_loss, epoch)

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch

            # 保存最佳模型
            best_model_file = os.path.join(opt.save_folder, 'best_model.pth')
            save_model(model, optimizer, opt, epoch, best_model_file)

            print(f' 新的最佳模型! Epoch: {epoch}, Loss: {loss:.4f}')

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    print(f"最佳模型: Epoch {best_epoch}, Loss: {best_loss:.4f}")
    summary_text = f"Training Summary:\nBest Epoch: {best_epoch}\nBest Loss: {best_loss:.4f}\nTotal Epochs: {opt.epochs}"



if __name__ == '__main__':
    main()
