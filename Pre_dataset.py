import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import random


class FaultDataset(Dataset):
    def __init__(self, data_dir, seq_length=1024, transform=None, normalize=True, mode='train',
                 max_samples_per_class=1024, seed=None, subset_mode='all', selection_seed=None):
        """
        Args:
            data_dir (str): 数据集根目录路径
            seq_length (int): 每个样本的序列长度 L
            transform (callable, optional): 可选的数据变换
            normalize (bool): 是否进行归一化
            mode (str): 'train' 或 'val'，决定使用1还是2文件夹
            max_samples_per_class (int): 每个类别的最大样本数，默认1024
            seed (int, optional): 随机种子，用于确定性采样
            subset_mode (str): 'all', 'first_half', 'second_half'. 
                               'first_half': 使用打乱后的前50%数据。
                               'second_half': 使用打乱后的后50%数据。
            selection_seed (int, optional): 用于在切分后再次随机选取的种子。
                                            如果不提供但提供了seed，将使用seed+1作为默认值以保证确定性。
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.transform = transform
        self.normalize = normalize
        self.mode = mode
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed
        self.subset_mode = subset_mode
        self.selection_seed = selection_seed

        # 故障类型映射
        self.fault_types = ['broken', 'healthy', 'missing_tooth', 'root_crack', 'wear']
        # 根据您提供的文件名，频率文件可能是 B1_20.MAT, B1_25.MAT 等格式
        self.freq_patterns = ['20', '25', '30', '35', '40', '45', '50', '55']

        self.data_list = []  # 存储所有分割后的序列
        self.label_list = []  # 存储对应的标签

        #print(f"初始化数据集: 模式={mode}, 数据目录={data_dir}")
        #print(f"每个类别限制样本数: {max_samples_per_class}")

        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        # 为每个类别单独收集样本
        class_samples = {i: [] for i in range(len(self.fault_types))}
        class_labels = {i: [] for i in range(len(self.fault_types))}

        # 遍历所有故障类型
        total_files_processed = 0
        for fault_idx, fault_type in enumerate(self.fault_types):
            fault_dir = os.path.join(data_dir, fault_type)
            if not os.path.exists(fault_dir):
                print(f"警告: 故障类型目录不存在: {fault_dir}")
                continue

            # 根据模式选择文件夹
            data_folder = os.path.join(fault_dir, '1' if mode == 'train' else '2')
            if not os.path.exists(data_folder):
                print(f"警告: 数据文件夹不存在: {data_folder}")
                continue

            print(f"处理故障类型: {fault_type} -> {data_folder}")

            # 获取文件夹中所有的MAT文件（支持 .mat 和 .MAT）
            all_mat_files = sorted([f for f in os.listdir(data_folder) if f.lower().endswith('.mat')])
            #print(f"  找到 {len(all_mat_files)} 个MAT文件: {all_mat_files}")

            # 根据故障类型和模式构建预期的文件名模式
            if fault_type == 'broken':
                prefix = 'B1' if mode == 'train' else 'B2'
            elif fault_type == 'healthy':
                prefix = 'N1' if mode == 'train' else 'N2'
            elif fault_type == 'missing_tooth':
                prefix = 'M1' if mode == 'train' else 'M2'
            elif fault_type == 'root_crack':
                prefix = 'R1' if mode == 'train' else 'R2'
            elif fault_type == 'wear':
                prefix = 'W1' if mode == 'train' else 'W2'
            else:
                prefix = ''

            # 遍历所有频率
            for freq in self.freq_patterns:
                # 构建可能的文件名模式
                expected_patterns = [
                    f"{prefix}_{freq}.MAT",
                    f"{prefix}_{freq}.mat",
                    f"{freq}.MAT",
                    f"{freq}.mat",
                    f"{freq}hz.MAT",
                    f"{freq}hz.mat"
                ]

                mat_file = None
                for pattern in expected_patterns:
                    potential_file = os.path.join(data_folder, pattern)
                    if os.path.exists(potential_file):
                        mat_file = potential_file
                        break

                if mat_file is None:
                    # 如果没有找到精确匹配，尝试在文件列表中查找包含频率的文件
                    matching_files = [f for f in all_mat_files if freq in f]
                    if matching_files:
                        mat_file = os.path.join(data_folder, matching_files[0])
                    else:
                        print(f"  警告: 未找到 {freq}Hz 的MAT文件")
                        continue

                try:
                    #print(f"  加载文件: {mat_file}")
                    # 加载mat文件
                    mat_data = sio.loadmat(mat_file)

                    # 查找数据变量（可能是Data、data或其他名称）
                    data_key = None
                    for key in mat_data.keys():
                        if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                            data_key = key
                            break

                    if data_key is None:
                        print(f"  错误: 在 {mat_file} 中找不到数据变量")
                        print(f"  可用变量: {list(mat_data.keys())}")
                        continue

                    data_array = mat_data[data_key]  # 获取数据数组
                    #print(f"  原始数据形状: {data_array.shape}, 变量名: {data_key}")

                    # 确保数据有4个通道，然后取前3个
                    if data_array.shape[1] >= 3:
                        data_array = data_array[:, :3].astype(np.float32)  # (N, 3)
                    else:
                        data_array = data_array.astype(np.float32)  # 如果不足3个通道，使用所有通道

                    #print(f"  处理后数据形状: {data_array.shape}")

                    # 如果启用normalize，进行逐通道归一化
                    if self.normalize:
                        #print("  进行归一化...")
                        for col in range(data_array.shape[1]):
                            scaler = StandardScaler()
                            data_array[:, col] = scaler.fit_transform(data_array[:, col].reshape(-1, 1)).flatten()

                    # 分割成固定长度序列
                    num_samples = len(data_array) // seq_length
                    #print(f"  可分割样本数: {num_samples}")

                    # 收集当前文件的样本
                    file_samples = []
                    for i in range(num_samples):
                        segment = data_array[i * seq_length: (i + 1) * seq_length]  # (L, 3)
                        file_samples.append(segment.transpose())  # 转置为 (3, L)

                    # 添加到类别样本中
                    class_samples[fault_idx].extend(file_samples)
                    class_labels[fault_idx].extend([fault_idx] * len(file_samples))

                    total_files_processed += 1
                    print(f"  当前类别 {fault_type} 累计样本数: {len(class_samples[fault_idx])}")

                except Exception as e:
                    print(f"  错误加载 {mat_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 限制每个类别的样本数量
        print("\n=== 限制每个类别样本数量 ===")
        for fault_idx, fault_type in enumerate(self.fault_types):
            samples_count = len(class_samples[fault_idx])
            if samples_count == 0:
                print(f"类别 {fault_type}: 无样本")
                continue

            # Determine indices based on seed and max_samples_per_class
            all_indices = list(range(samples_count))
            
            if self.seed is not None:
                rng = random.Random(self.seed)
                rng.shuffle(all_indices)
            else:
                # If no seed, use global random state
                random.shuffle(all_indices)
            
            # Apply subset split
            if self.subset_mode == 'first_half':
                split_point = samples_count // 2
                available_indices = all_indices[:split_point]
                print(f"类别 {fault_type}: 选取前 50% ({len(available_indices)}/{samples_count})")
            elif self.subset_mode == 'second_half':
                split_point = samples_count // 2
                available_indices = all_indices[split_point:]
                print(f"类别 {fault_type}: 选取后 50% ({len(available_indices)}/{samples_count})")
            else:
                available_indices = all_indices

            # Apply max_samples_per_class limit
            if len(available_indices) > self.max_samples_per_class:
                # 在切分好的数据中再随机选取，使用独立的种子
                current_selection_seed = self.selection_seed
                
                # 如果没有指定 selection_seed 但指定了全局 seed，为了保证确定性且使用不同的随机流，
                # 我们可以派生一个新的种子
                if current_selection_seed is None and self.seed is not None:
                    current_selection_seed = self.seed + 1
                
                if current_selection_seed is not None:
                    rng_selection = random.Random(current_selection_seed)
                    rng_selection.shuffle(available_indices)
                    print(f"类别 {fault_type}: 使用独立种子 {current_selection_seed} 进行二次随机选取")
                else:
                    random.shuffle(available_indices)
                    print(f"类别 {fault_type}: 使用全局随机状态进行二次随机选取")
                    
                selected_indices = available_indices[:self.max_samples_per_class]
                print(f"类别 {fault_type}: 限制样本数至 {self.max_samples_per_class} (随机选取)")
            else:
                selected_indices = available_indices
                print(f"类别 {fault_type}: 使用所有可用样本 {len(selected_indices)}")

            selected_samples = [class_samples[fault_idx][i] for i in selected_indices]
            selected_labels = [class_labels[fault_idx][i] for i in selected_indices]

            self.data_list.extend(selected_samples)
            self.label_list.extend(selected_labels)

        print(f"\n最终数据集统计:")
        print(f"总样本数: {len(self.data_list)}")
        for fault_idx, fault_type in enumerate(self.fault_types):
            count = sum(1 for label in self.label_list if label == fault_idx)
            print(f"类别 {fault_type}: {count} 个样本")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        segment = self.data_list[idx]  # (3, L)
        label = self.label_list[idx]

        # 转换为Tensor
        segment = torch.from_numpy(segment).float()  # (3, L)

        # 可选变换
        if self.transform:
            segment = self.transform(segment)

        return segment, label

    def get_dataset_info(self):
        """返回数据集的详细信息"""
        if len(self.data_list) == 0:
            return "数据集为空"

        # 计算各种统计信息
        total_samples = len(self.data_list)
        seq_length = self.data_list[0].shape[1]
        num_channels = self.data_list[0].shape[0]

        # 计算总数据量
        total_elements = sum(data.size for data in self.data_list)
        total_memory_mb = (total_elements * 4) / (1024 * 1024)  # float32，4字节

        # 标签分布
        label_counts = np.bincount(self.label_list)
        label_distribution = {self.fault_types[i]: count for i, count in enumerate(label_counts)}

        info = f"""
=== 数据集详细信息 ===
模式: {self.mode}
总样本数: {total_samples}
序列长度: {seq_length}
通道数: {num_channels}
数据形状: ({num_channels}, {seq_length})
总数据量: {total_elements} 个元素, 约 {total_memory_mb:.2f} MB
标签分布: {label_distribution}
        """
        return info


# 示例使用函数：创建DataLoader
def get_dataloader(data_dir, seq_length=1024, batch_size=32, shuffle=True,
                   num_workers=4, normalize=True, mode='train', max_samples_per_class=1024):
    dataset = FaultDataset(data_dir, seq_length=seq_length, normalize=normalize,
                           mode=mode, max_samples_per_class=max_samples_per_class)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)
    return dataloader, dataset


# 测试代码
if __name__ == "__main__":
    data_dir = "/root/autodl-fs"
    seq_length = 1024

    try:
        # 创建训练数据集
        print("=== 创建训练数据集 ===")
        dataset_train = FaultDataset(data_dir, seq_length=seq_length, normalize=True,
                                     mode='train', max_samples_per_class=1024)
        print(dataset_train.get_dataset_info())

        # 创建验证数据集
        print("\n=== 创建验证数据集 ===")
        dataset_val = FaultDataset(data_dir, seq_length=seq_length, normalize=True,
                                   mode='val', max_samples_per_class=1024)
        print(dataset_val.get_dataset_info())

        if len(dataset_train) == 0 or len(dataset_val) == 0:
            print("数据集为空，请检查数据路径和文件。")
        else:
            print(f"\n=== 汇总信息 ===")
            print(f"训练集样本数: {len(dataset_train)}")
            print(f"验证集样本数: {len(dataset_val)}")
            print(f"总样本数: {len(dataset_train) + len(dataset_val)}")

            # 测试一个样本
            sample, label = dataset_train[0]
            print(f"\n单个样本信息:")
            print(f"样本形状: {sample.shape}")
            print(f"数据类型: {sample.dtype}")
            print(f"标签: {label} (对应故障类型: {dataset_train.fault_types[label]})")

            # 显示数据范围
            print(
                f"数据范围: min={sample.min():.4f}, max={sample.max():.4f}, mean={sample.mean():.4f}, std={sample.std():.4f}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
