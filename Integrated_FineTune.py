from __future__ import print_function

import os
import sys
import argparse
import time
import math
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix # 导入新的评估指标

from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter  # 新增导入
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import weakCustomTransform,strongCustomTransform
from util import save_model
from util import plot_confusion_matrix, get_features, visualize_tsne
from util import TwoCropTransform,TransformedSubset
from networks.swmodel import  ContMixDualStream,ContMixContrastive,LinearClassifier
from Pre_dataset import  FaultDataset
from print_data import extract_labels
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # transformations parameters（与 Pre_train 对齐）
    parser.add_argument('--scale_mu', type=float, default=2.0, help='mu for gaussian scaling in weak transform')
    parser.add_argument('--scale_sigma', type=float, default=1.1, help='sigma for gaussian scaling in weak transform')
    parser.add_argument('--jitter_loc', type=float, default=0.0, help='loc for jitter noise in weak transform')
    parser.add_argument('--jitter_sigma', type=float, default=0.8, help='sigma for jitter noise in weak transform')
    parser.add_argument('--trans_max_shift', type=int, default=3, help='max circular shift for translation in strong transform')
    parser.add_argument('--snr_db', type=float, default=6.0, help='SNR (dB) for additive noise in strong transform')

    # model dataset
    parser.add_argument('--seq_length', type=int, default=1024,
                        help='sequence length for the dataset')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model type (e.g., stfencoder)')
    parser.add_argument('--dataset', type=str, default='custom',
                        choices=['cifar10', 'cifar100', 'custom'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='/root/autodl-fs', help='path to custom dataset directory')
    parser.add_argument('--normalize', action='store_true', help='enable normalization in dataset')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--supcon_weight', type=float, default=0.1,
                        help='weight for supervised contrastive loss')
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    parser.add_argument('--perplexity', type=float, default=30.0, help='perplexity for t-SNE')
    
    # Dataset Pool Limit (Should be large enough to cover Train + Val)
    parser.add_argument('--max_samples_per_class', type=int, default=2000, help='max pool samples per class')
    # Actual Training and Validation Set Sizes
    parser.add_argument('--num_train_per_class', type=int, default=50, help='number of training samples per class')
    parser.add_argument('--num_val_per_class', type=int, default=300, help='number of validation samples per class')

    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = '/root/autodl-fs' # Now passed via argument
    opt.save_folder = './save/FineTune_models' # Updated folder name match shell script expectation approximately
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # Name model based on TRAINING samples count, not pool size
    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_samples_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial, opt.num_train_per_class)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'custom':
        opt.n_cls = 5

    return opt

def set_loader(opt):
    # Data Augmentation for Fine-Tuning
    # Typically only weak aug or standard aug for fine-tuning
    
    # Using 'val' mode folder as per user request history, or 'train'
    # Defaulting to 'val' folder (folder 2) as before if that has enough data.
    # But usually pre-training uses 'train' folder.
    # Let's use 'val' folder for supervised task consistency with previous WDCNN task.
    train_transform = None
    val_transform = None

    full_dataset = FaultDataset(
        data_dir=opt.data_folder,
        seq_length=opt.seq_length,
        transform=None,
        mode='train', 
        normalize=opt.normalize,
        max_samples_per_class=opt.max_samples_per_class,
        seed=42,
        subset_mode='second_half' 
    )

    labels = np.array(full_dataset.label_list)
    unique_labels = np.unique(labels)
    
    train_indices = []
    val_indices = []

    # Manual split to ensure no leakage
    for cls in unique_labels:
        cls_indices = np.where(labels == cls)[0]
        
        # Sort and Shuffle deterministic
        cls_indices = sorted(cls_indices)
        np.random.seed(42)
        np.random.shuffle(cls_indices)
        
        train_cls_indices = cls_indices[:opt.num_train_per_class]
        val_cls_indices = cls_indices[opt.num_train_per_class : opt.num_train_per_class + opt.num_val_per_class]
        
        train_indices.extend(train_cls_indices)
        val_indices.extend(val_cls_indices)
        
    print(f"FineTune Train Samples: {len(train_indices)}")
    print(f"FineTune Val   Samples: {len(val_indices)}")
    
    # Add Augmentation to Training Set if needed   
    train_dataset = TransformedSubset(full_dataset, train_indices, transform=train_transform)
    val_dataset = TransformedSubset(full_dataset, val_indices, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )

    return train_loader, val_loader

def set_model(opt):
    encoder = ContMixDualStream(
        in_channels=3,
        embed_dim=128,
        patch_size=8,
        depth=3,
        kernel_sizes=[7, 15, 31],
        num_heads=8,
        num_classes=None
    )
    model = ContMixContrastive(encoder=encoder, proj_dim=64)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=opt.temp)

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        criterion_supcon = criterion_supcon.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion, criterion_supcon

def train(train_loader, model, classifier, criterion, criterion_supcon, optimizer, epoch, opt,writer=None):
    """one epoch training"""
    # 微调模式：Encoder 和 Classifier 都开启训练模式
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (batch_data, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if isinstance(batch_data, list) and len(batch_data) == 2:
            # TwoCropTransform返回 [view1, view2]，我们只需要一个视图用于分类训练
            images = batch_data[1]  # 使用第一个视图（弱增强）
        else:
            images = batch_data           

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        # 微调模式：移除 torch.no_grad()，允许梯度回传给 Encoder
        features = model.encoder(images)
        output = classifier(features)
        
        # CE Loss
        loss_ce = criterion(output, labels)
        
        # SupCon Loss (using features before classifier)
        # SupCon expects features of shape [bsz, n_views, dim]
        # Here we only have 1 view, so we unsqueeze
        features_supcon = features.unsqueeze(1)
        # Normalize features for SupCon
        features_supcon = torch.nn.functional.normalize(features_supcon, dim=2)
        loss_supcon = criterion_supcon(features_supcon, labels)
        
        # Total Loss
        loss = loss_ce + opt.supcon_weight * loss_supcon

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
        if writer is not None:
            writer.add_scalar("Train/Loss", losses.avg, epoch)
            writer.add_scalar("Train/Accuracy", top1.avg, epoch)
    return losses.avg, top1.avg


def validate(val_loader, model,classifier, criterion, opt,writer=None, epoch=0):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        end = time.time()
        for idx, (batch_data, labels) in enumerate(val_loader):
            if isinstance(batch_data, list) and len(batch_data) == 2:
            # TwoCropTransform返回 [view1, view2]，我们只需要一个视图用于分类训练
                images = batch_data[1]  # 使用第一个视图（弱增强）
            else:
                images = batch_data

            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    #cm = confusion_matrix(all_labels, all_preds)

    print(' * Acc@1 {top1.avg:.3f}, F1 {f1:.3f}, Precision {precision:.3f}, Recall {recall:.3f}'.format(
        top1=top1, f1=f1, precision=precision, recall=recall))
    if writer is not None:
        writer.add_scalar("Val/Loss", losses.avg, epoch)
        writer.add_scalar("Val/Accuracy", top1.avg, epoch)
        writer.add_scalar("Val/F1", f1, epoch)
        writer.add_scalar("Val/Precision", precision, epoch)
        writer.add_scalar("Val/Recall", recall, epoch)
        for i, score in enumerate(f1_per_class):
            writer.add_scalar(f"Val/F1_Class_{i}", score, epoch)
    return losses.avg, top1.avg,all_preds, all_labels

def main():
    best_acc = 0
    best_epoch = 0
    best_preds, best_labels = None, None
    opt = parse_option()


    # 构建数据加载器
    train_loader, val_loader = set_loader(opt)

    # 从训练集和验证集中提取标签
    train_labels = extract_labels(train_loader)
    val_labels = extract_labels(val_loader)

    # 可视化训练集和验证集的标签分布
    #plot_class_distribution(train_labels, title="Training Set Class Distribution")
    #plot_class_distribution(val_labels, title="Validation Set Class Distribution")
    # 构建模型和损失函数
    model,classifier,criterion, criterion_supcon = set_model(opt)

    # 构建优化器 (微调模式：同时优化 Encoder 和 Classifier)
    parameters = list(model.encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.SGD(parameters,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # TensorBoard
    writer = SummaryWriter(log_dir=f'./runs/{opt.model_name}_finetune')

    # Store original supcon weight
    original_supcon_weight = opt.supcon_weight

    # 训练过程
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # Dynamic SupCon Strategy:
        # First 20% epochs: use full weight to cluster features
        # After 20% epochs: set weight to 0 to refine classification boundaries
        if epoch <= int(opt.epochs * 0.2):
            opt.supcon_weight = original_supcon_weight
        else:
            opt.supcon_weight = 0.0
        
        print(f"Epoch {epoch}: Current SupCon Weight: {opt.supcon_weight}")

        # 一个epoch训练
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, classifier, criterion, criterion_supcon,
                          optimizer, epoch, opt,writer=writer)
        time2 = time.time()
        print('************************************************************')
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, train_acc))

        time3 = time.time()
        loss, val_acc,preds, labels = validate(val_loader, model, classifier, criterion, opt, writer=writer, epoch=epoch)
        time4 = time.time()
        print('Test epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time4 - time3, val_acc))
        print('------------------------------------------------------------')
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_preds, best_labels = preds, labels
            best_model_state = model.state_dict()
            
            # Save best model
            save_file = os.path.join(opt.save_folder, f'{opt.model_name}_best.pth')
            save_model(model, optimizer, opt, epoch, save_file, classifier=classifier)
            print(f'Best model saved to {save_file}')

    if best_preds is not None:
        cm = confusion_matrix(best_labels, best_preds)
        fig = plot_confusion_matrix(cm, classes=[str(i) for i in range(opt.n_cls)],
                                    writer=writer,
                                    epoch=best_epoch,
                                    normalize=False,
                                  title='Best Confusion Matrix')
        plt.close(fig)

        # t-SNE Visualization
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loading best model (Epoch {best_epoch}) for t-SNE visualization...")
            features, labels = get_features(val_loader, model, device='cuda' if torch.cuda.is_available() else 'cpu')
            tsne_save_path = f'./runs/{opt.model_name}/tsne_best_epoch_{best_epoch}.png'
            # Ensure directory exists
            os.makedirs(os.path.dirname(tsne_save_path), exist_ok=True)
            visualize_tsne(features, labels, opt.n_cls, save_path=tsne_save_path, perplexity=opt.perplexity)

    writer.close()

if __name__ == '__main__':
    main()
