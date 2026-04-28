import random
from typing import List

from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import itertools
import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data.dataset import T_co


# ---------- 3.1 物理一致增广 ----------
def jitter(x, sigma=0.02):
    if x.dim() == 2:  # 单个样本 (7, L)
        std = x.std(dim=1, keepdim=True)  # (7, 1)
        noise = sigma * std * torch.randn_like(x)
    else:  # batch 样本 (B, 7, L)
        std = x.std(dim=2, keepdim=True)  # (B, 7, 1)
        noise = sigma * std * torch.randn_like(x)
    return x + noise

def scaling(x, low=0.9, high=1.1):
    if x.dim() == 2:  # 单个样本 (7, L)
        factor = torch.empty(x.size(0), 1, device=x.device).uniform_(low, high)
    else:  # batch 样本 (B, 7, L)
        factor = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(low, high)
    return x * factor

def time_shift(x, max_shift=3):
    if x.dim() == 2:  # 单个样本 (7, L)
        shift = torch.randint(-max_shift, max_shift+1, (x.size(0),), device=x.device)
        out = torch.zeros_like(x)
        for i, s in enumerate(shift):
            if s >= 0:
                out[i, s:] = x[i, :-s] if s > 0 else x[i]
            else:
                out[i, :s] = x[i, -s:]
    else:  # batch 样本 (B, 7, L)
        shift = torch.randint(-max_shift, max_shift+1, (x.size(0),), device=x.device)
        out = torch.zeros_like(x)
        for b in range(x.size(0)):
            s = shift[b]
            if s >= 0:
                out[b, :, s:] = x[b, :, :-s] if s > 0 else x[b]
            else:
                out[b, :, :s] = x[b, :, -s:]
    return out


def sensor_lag(x, tau_range=(0.1, 0.5), fs=1.0):
    # 确定输入维度
    if x.dim() == 2:  # 单个样本 (7, L)
        tau = torch.empty(x.size(0), 1, device=x.device).uniform_(*tau_range)
        alpha = torch.exp(-1.0 / (tau * fs))
        # 移除最后一个维度，使其形状为 (7,)
        alpha = alpha.squeeze(-1)
        y = x.clone()
        for t in range(1, x.size(-1)):
            y[:, t] = alpha * y[:, t - 1] + (1 - alpha) * x[:, t]

    else:  # batch 样本 (B, 7, L)
        tau = torch.empty(x.size(0), x.size(1), 1, device=x.device).uniform_(*tau_range)
        alpha = torch.exp(-1.0 / (tau * fs))
        # 移除最后一个维度，使其形状为 (B, 7)
        alpha = alpha.squeeze(-1)
        y = x.clone()
        for t in range(1, x.size(-1)):
            y[:, :, t] = alpha * y[:, :, t - 1] + (1 - alpha) * x[:, :, t]

    return y


# ---------- 基于你的公式的增强实现 ----------
def gaussian_scaling(x, mu=2.0, sigma=0.1):
    # \bar{x} = x * s_c, s_c ~ N(mu, sigma)
    if x.dim() == 2:  # (C, L)
        s = torch.normal(mean=torch.full((x.size(0), 1), mu, device=x.device, dtype=x.dtype),
                         std=torch.full((x.size(0), 1), sigma, device=x.device, dtype=x.dtype))
    else:  # (B, C, L)
        s = torch.normal(mean=torch.full((x.size(0), x.size(1), 1), mu, device=x.device, dtype=x.dtype),
                         std=torch.full((x.size(0), x.size(1), 1), sigma, device=x.device, dtype=x.dtype))
    s = torch.clamp(s, min=1e-6)
    return x * s

def jitter_gaussian(x, loc=0.0, sigma=0.02):
    # \bar{x} = x + N(loc, sigma)
    noise = torch.normal(mean=loc, std=sigma, size=x.shape, device=x.device, dtype=x.dtype)
    return x + noise

def translate_circular(x, max_shift=3, shift=None):
    # \bar{x} = concat(x[p:end], x[0:p-1])
    if shift is None:
        if x.dim() == 2:
            s = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item())
            return torch.roll(x, shifts=s, dims=-1)
        else:
            B = x.size(0)
            out = x.clone()
            shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
            for b in range(B):
                out[b] = torch.roll(x[b], shifts=int(shifts[b].item()), dims=-1)
            return out
    else:
        return torch.roll(x, shifts=int(shift), dims=-1)

def add_noise_snr(x, snr_db=20.0):
    # \bar{x} = x + G_n(x, SNR)
    eps = 1e-12
    if x.dim() == 2:  # (C, L)
        Ps = (x.pow(2).mean(dim=-1, keepdim=True)).clamp(min=eps)  # (C,1)
        snr_lin = 10 ** (snr_db / 10.0)
        Pn = Ps / snr_lin
        std = Pn.sqrt()
        noise = torch.randn_like(x) * std
        return x + noise
    else:  # (B, C, L)
        Ps = (x.pow(2).mean(dim=-1, keepdim=True)).clamp(min=eps)  # (B,C,1)
        snr_lin = 10 ** (snr_db / 10.0)
        Pn = Ps / snr_lin
        std = Pn.sqrt()
        noise = torch.randn_like(x) * std
        return x + noise


# ---------- 用上述四种增强重写 weak/strong ----------
class weakCustomTransform:
    def __init__(self, mu=1.0, delta_s=0.1, loc=0.0, jitter_sigma=0.02):
        self.mu = mu
        self.delta_s = delta_s
        self.loc = loc
        self.jitter_sigma = jitter_sigma

    def __call__(self, x):
        x = gaussian_scaling(x, mu=self.mu, sigma=self.delta_s)
        x = jitter_gaussian(x, loc=self.loc, sigma=self.jitter_sigma)
        return x

class strongCustomTransform:
    def __init__(self, max_shift=3, snr_db=20.0):
        self.max_shift = max_shift
        self.snr_db = snr_db

    def __call__(self, x):
        x = translate_circular(x, max_shift=self.max_shift)
        x = add_noise_snr(x, snr_db=self.snr_db)
        return x

class finetuningTransform:
    def __init__(self, max_shift=1, snr_db=30.0):
        # 相比 strongCustomTransform，位移更小 (max_shift 3->1)，信噪比更高 (snr_db 20->30，即噪声更小)
        self.max_shift = max_shift
        self.snr_db = snr_db

    def __call__(self, x):
        x = translate_circular(x, max_shift=self.max_shift)
        x = add_noise_snr(x, snr_db=self.snr_db)
        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file, classifier=None):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if classifier is not None:
        state['classifier'] = classifier.state_dict()
        
    torch.save(state, save_file)
    del state

def plot_confusion_matrix(cm, classes, writer, epoch, normalize=False, title='Confusion matrix'):
    """绘制混淆矩阵并写入 TensorBoard"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 写数字
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    writer.add_figure("Confusion_matrix", fig, epoch)
    plt.close(fig)

def train_transform(x):
    if random.random() < 0.5:
        return strongCustomTransform(x)
    else:
        return weakCustomTransform(x)

def val_transform(x):
    return None
class TransformedSubset(torch.utils.data.Subset):
    def __init__(self,dataset,indices,transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
    def __getitem__(self,idx):
        data,label = super().__getitem__(idx)
        if self.transform is not None:
            view1, view2 = self.transform(data)
            return (view1, view2), label
        else:
            return data,label
class TwoCropTransform:
    """Create two crops of the same sample using new weak/strong transforms"""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        return [self.weak_transform(x), self.strong_transform(x)]


def visualize_time_series(view1, view2, label, writer, global_step):
    """生成时间序列可视化图"""
    # 设置通道名称
    channel_names = ['FT1', 'FP1', 'FP4', 'FGf', 'FP6', 'FT6', 'Pe_PT']

    # 创建一个大图表，包含所有通道
    fig, axes = plt.subplots(7, 2, figsize=(14, 18))
    fig.suptitle(f'Sample Visualization (Label: {label})', fontsize=16)

    # 绘制每个通道
    for i, name in enumerate(channel_names):
        # 原始数据
        signal1 = view1[i].cpu().numpy()
        axes[i, 0].plot(signal1, 'b-')
        axes[i, 0].set_title(f'{name} - Weak View')
        axes[i, 0].set_ylabel('Amplitude')

        # 增强后的数据
        signal2 = view2[i].cpu().numpy()
        axes[i, 1].plot(signal2, 'r-')
        axes[i, 1].set_title(f'{name} - Strong View')

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 转换为图像并添加到TensorBoard
    writer.add_figure(f'Sample/time_series', fig, global_step)
    plt.close(fig)


def visualize_frequency_domain(view1, view2, label, writer, global_step):
    """生成频域可视化图"""
    # 设置通道名称
    channel_names = ['FT1', 'FP1', 'FP4', 'FGf', 'FP6', 'FT6', 'Pe_PT']

    # 创建一个大图表，包含所有通道
    fig, axes = plt.subplots(7, 2, figsize=(14, 18))
    fig.suptitle(f'Frequency Domain Visualization (Label: {label})', fontsize=16)

    # 计算并绘制每个通道的频谱
    for i, name in enumerate(channel_names):
        # 原始数据频谱
        signal1 = view1[i].cpu().numpy()
        fft1 = np.abs(np.fft.rfft(signal1))
        freq1 = np.fft.rfftfreq(len(signal1))
        axes[i, 0].plot(freq1, fft1, 'b-')
        axes[i, 0].set_title(f'{name} - Weak View (Freq)')
        axes[i, 0].set_ylabel('Magnitude')
        axes[i, 0].set_xlabel('Frequency')

        # 增强后的数据频谱
        signal2 = view2[i].cpu().numpy()
        fft2 = np.abs(np.fft.rfft(signal2))
        freq2 = np.fft.rfftfreq(len(signal2))
        axes[i, 1].plot(freq2, fft2, 'r-')
        axes[i, 1].set_title(f'{name} - Strong View (Freq)')
        axes[i, 1].set_xlabel('Frequency')

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 转换为图像并添加到TensorBoard
    writer.add_figure(f'Sample/frequency_domain', fig, global_step)
    plt.close(fig)


def log_embedding_visualization(model, train_loader, writer, epoch):
    """使用t-SNE或PCA可视化特征嵌入"""
    print("生成嵌入可视化...")
    model.eval()
    features_list = []
    labels_list = []

    # 仅使用少量批次以加快处理速度
    max_batches = 5
    with torch.no_grad():
        for idx, (views, labels) in enumerate(train_loader):
            if idx >= max_batches:
                break

            if torch.cuda.is_available():
                views = [view.cuda() for view in views]

            # 获取原始视图的特征
            features = model(views[0])  # 只使用weak视图
            features_list.append(features.cpu())
            labels_list.append(labels)

    # 合并所有特征和标签
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # 使用PCA降维到2D进行可视化
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.numpy())

    # 添加到TensorBoard
    writer.add_embedding(
        features_2d,
        metadata=labels.numpy(),
        global_step=epoch,
        tag='feature_embeddings'
    )
    print("嵌入可视化完成")

def get_features(loader, model, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for idx, (batch_data, target) in enumerate(loader):
            if isinstance(batch_data, list) and len(batch_data) == 2:
                images = batch_data[1]
            else:
                images = batch_data
            
            images = images.float()
            if device:
                images = images.to(device)
            
            # Check if model is ContMixContrastive or just the encoder
            if hasattr(model, 'encoder'):
                feat = model.encoder(images)
            else:
                feat = model(images)
                
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
            
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def visualize_tsne(
    features,
    labels,
    num_classes,
    save_path=None,
    title="t-SNE Visualization",
    perplexity=30,
    n_components=2,
    pca_dim=None,
    normalize_features=True,
    metric="euclidean",
    init="pca",
    random_state=0,
    early_exaggeration=12.0,
    learning_rate="auto",
    max_iter=2000,
):
    """t-SNE visualization with optional PCA pre-reduction.

    Args:
        features: (N, D) array-like or torch.Tensor.
        labels: (N,) array-like or torch.Tensor.
        num_classes: number of classes for coloring.
        save_path: if provided, saves the figure.
        title: plot title.
        perplexity: t-SNE perplexity.
        n_components: output dimension of t-SNE (2 or 3).
        pca_dim: if set (e.g., 50), applies PCA to reduce features to pca_dim before t-SNE.
        normalize_features: if True, L2-normalize each feature vector before PCA/t-SNE.
        metric: distance metric for t-SNE (e.g., 'euclidean' or 'cosine').
    """

    print(
        "Computing t-SNE with "
        f"perplexity={perplexity}, n_components={n_components}, pca_dim={pca_dim}, "
        f"normalize_features={normalize_features}, metric={metric}, init={init}, "
        f"early_exaggeration={early_exaggeration}, learning_rate={learning_rate}, max_iter={max_iter}..."
    )
    # Set font to serif (will try Times New Roman if available, else fallback)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    else:
        features = np.asarray(features)

    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    if normalize_features:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-12)

    features_for_tsne = features
    if pca_dim is not None:
        pca_dim = int(pca_dim)
        if pca_dim < n_components:
            raise ValueError(
                f"pca_dim must be >= n_components (got pca_dim={pca_dim}, n_components={n_components})"
            )
        pca_dim = min(pca_dim, features.shape[1])
        if pca_dim < features.shape[1]:
            print(f"Applying PCA pre-reduction: {features.shape[1]} -> {pca_dim} dims")
            pca = PCA(n_components=pca_dim, random_state=0)
            features_for_tsne = pca.fit_transform(features)

    from sklearn.manifold import TSNE
    tsne_kwargs = dict(
        n_components=n_components,
        init=init,
        random_state=random_state,
        perplexity=perplexity,
        metric=metric,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
    )
    # sklearn 的参数名随版本变化：有的叫 max_iter，有的叫 n_iter
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=max_iter)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=max_iter)
    features_tsne = tsne.fit_transform(features_for_tsne)
    
    plt.figure(figsize=(10, 8))
    # Use a colormap that can handle the number of classes
    cmap = plt.get_cmap('tab10') if num_classes <= 10 else plt.get_cmap('tab20')
    
    if n_components == 3:
        ax = plt.axes(projection='3d')
        for i in range(num_classes):
            indices = labels == i
            ax.scatter3D(features_tsne[indices, 0], features_tsne[indices, 1], features_tsne[indices, 2],
                        color=cmap(i % cmap.N), label=str(i), alpha=0.6, s=20)
    else:
        for i in range(num_classes):
            indices = labels == i
            plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], 
                        color=cmap(i % cmap.N), label=str(i), alpha=0.6, s=20)
                    
    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(title)
    
    # Keep ticks but remove labels
    if n_components == 2:
        plt.xlabel('')
        plt.ylabel('')
    
    if save_path:
        # Use bbox_inches='tight' to include the outside legend
        plt.savefig(save_path, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
        plt.close()