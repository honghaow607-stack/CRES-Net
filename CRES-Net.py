import os
import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle

BATCH_SIZE = 256  # batch size
TRAIN_EPOCHS = 100  # number of epoch
FOLD = 5  # = NUM_SAMPLE / number of val samples
Num_class = 2
result_path = 'path/to/checkpoints'
LAMDA = 0.05 # temperature 超参
LR = 0.002 # learning rate
BN_DIM = 300 # batch normalization dimension 
TRAIN_LR = 3e-4
WEIGHT_DECAY = 3e-4
FOCAL_GAMMA = 1.0
NATIVE_WEIGHT = 0.12
ORDER_WEIGHT = 0.10
ORDER_MARGIN = 0.10
CROP_PROBABILITY = 0.50
CROP_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.80, 0.90, 1.00)
VALIDATE_EVERY = 2
EARLY_STOP_PATIENCE = 10
TRAIN_SEED = 20261018

def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list

def parse_sample(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(file_path):
        print(f"文件路径不存在：{file_path}")
        return None
    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:
        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample

def save_variable(file_name, variable):
    # 将变量保存到文件中
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()

def get_alter_loaders():

    File_Embed = "path/to/g729a_Steg_QIM_feat"
    File_NoEmbed = "path/to/g729a_0_QIM_feat"
    pklfilex = 'path/to/MSCRE_QIM.pkl'

    if not os.path.exists(pklfilex):
        df = pd.read_csv('/root/autodl-tmp/data/data_SFFN/data_SFFN_train/data_SFFN_7_dim/train_lable.csv', header=None)
        file_list_embed = os.listdir(File_Embed)
        file_classes_embed = {}  # 存储嵌入文件的文件名和类别信息
        for file in file_list_embed:
            file_name = file.split('_')[0]  # 提取文件名
            matching_row = df.loc[df[0] == file_name]
            if not matching_row.empty:
                class_name = matching_row.iloc[0, 2]  # Retrieve the category information corresponding to the file
                if file not in file_classes_embed:  # 检查文件是否已经在字典中
                    file_classes_embed[file] = class_name  # 将文件名和类别信息添加到字典中
            
                

        # 按类别划分文件列表
        class_files_embed = {}
        for file, class_name in file_classes_embed.items():
            if class_name not in class_files_embed:
                class_files_embed[class_name] = []
            class_files_embed[class_name].append(file)

        file_list_noembed = os.listdir(File_NoEmbed)
        file_classes_noembed = {}  # 存储非嵌入文件的文件名和类别信息
        for file in file_list_noembed:
            # 为非嵌入文件赋予特殊类别标识，这里选择用-1表示
            class_name = -1
            file_classes_noembed[file] = class_name  # 将文件名和类别信息添加到字典中

        # 按类别划分文件列表
        class_files_noembed = {}
        for file, class_name in file_classes_noembed.items():
            if class_name not in class_files_noembed:
                class_files_noembed[class_name] = []
            class_files_noembed[class_name].append(file)

        # 划分测试集和训练集
        train, val = [], []
        # 遍历 class_files 字典，按 class_name 分别处理
        for class_name, files in class_files_embed.items():
            random.shuffle(files)  # 随机打乱文件顺序
            val_size = int(len(files) / FOLD)  # 计算测试集大小
            val_files = files[:val_size]  # 获取测试集文件列表
            train_files = files[val_size:]  # 获取训练集文件列表

            # 将测试集文件列表添加到 x_val 和 y_val 中
            for file in val_files:
                val.append([parse_sample(File_Embed, file), class_name, 1])
            # 将训练集文件列表添加到 x_train 和 y_train 中
            for file in train_files:
                train.append([parse_sample(File_Embed, file), class_name, 1])

        for class_name, files in class_files_noembed.items():
            val_size = int(len(files) / FOLD)  # 计算测试集大小

            # 对于非嵌入文件，按照 8:2 的比例划分为训练集和测试集
            val_files = files[:val_size]  # 获取测试集文件列表
            train_files = files[val_size:]  # 获取训练集文件列表

            # 将测试集文件列表添加到 x_val 和 y_val 中
            for file in val_files:
                val.append([parse_sample(File_NoEmbed, file), 0, 0])

            # 将训练集文件列表添加到 x_train 和 y_train 中
            for file in train_files:
                train.append([parse_sample(File_NoEmbed, file), 0, 0])

        random.shuffle(train)
        random.shuffle(val)

        x_train, y_train, x_val, y_val = [], [], [], []

        # 将train数据集拆分为x_train和y_train
        for sample in train:
            x_train.append(sample[0])  # 添加样本特征到x_train
            y_train.append(sample[1:])  # 添加标签到y_train

        # 将val数据集拆分为x_val和y_val
        for sample in val:
            x_val.append(sample[0])  # 添加样本特征到x_val
            y_val.append(sample[1:])  # 添加标签到y_val

        # 转换为numpy数组
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # 将数据保存到.pkl文件中
        with open(pklfilex, 'wb') as f:
            pickle.dump((x_train, y_train, x_val, y_val), f)

    else:
        # 如果.pkl文件存在，则直接加载数据
        with open(pklfilex, 'rb') as f:
            x_train, y_train, x_val, y_val = pickle.load(f)
    return x_train, y_train, x_val, y_val

def convert_to_loader_CL(x_train, y_train, x_val, y_val, batch_size):
    # 将numpy数组转换为PyTorch的Tensor
    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)
    x_val_tensor = torch.Tensor(x_val)
    y_val_tensor = torch.Tensor(y_val)

    # 创建训练集和测试集的数据集对象
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    # 获取标签为1的索引
    steg_indices = [i for i, label in enumerate(y_train) if label[1] == 1]
    # 获取标签为0的索引
    cover_indices = [i for i, label in enumerate(y_train) if label[1] == 0]

    # 创建标签为1的样本子集
    train_steg_dataset = Subset(train_dataset, steg_indices)
    # 创建标签为0的样本子集
    train_cover_dataset = Subset(train_dataset, cover_indices)

    train_steg_loader = DataLoader(train_steg_dataset, batch_size=batch_size, shuffle=True)
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 返回创建的DataLoader
    return train_steg_loader, train_cover_loader, val_loader

Num_layers = 3
FEATURE_COLUMNS = (0,1,2)  
HIDDEN_DIM = 96
EMBEDDING_DIM = 24


class TemporalResidualBlock(nn.Module):
    """膨胀卷积。层数增加时感受野自然覆盖不同时间尺度。"""

    def __init__(self, hidden_dim, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(0.12)

    def forward(self, x):
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + self.dropout(x)


class MSCRENet(nn.Module):
    """Multi-Scale Cover-Referenced Evidence Network。

    模型不知道算法、嵌入率和语音时长。它只学习：
    1. Cover条件下码字应该是什么；
    2. 观察码字与Cover期望之间的残差；
    3. 稀疏与弥散证据在不同时长下如何聚合。
    """

    FIELD_MIN = (0, 0, 0, 20, 20, -1, -1)
    FIELD_MAX = (127, 31, 31, 143, 143, 1, 1)

    def __init__(
        self,
        num_layers=Num_layers,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        feature_columns=FEATURE_COLUMNS,
    ):
        super().__init__()
        self.feature_columns = tuple(int(index) for index in feature_columns)
        if not self.feature_columns:
            raise ValueError("FEATURE_COLUMNS不能为空")
        if min(self.feature_columns) < 0 or max(self.feature_columns) > 6:
            raise ValueError("FEATURE_COLUMNS必须落在0..6")

        self.field_count = len(self.feature_columns)
        field_min = torch.tensor(
            [self.FIELD_MIN[index] for index in self.feature_columns],
            dtype=torch.long,
        )
        field_max = torch.tensor(
            [self.FIELD_MAX[index] for index in self.feature_columns],
            dtype=torch.long,
        )
        self.register_buffer('field_min', field_min)
        self.register_buffer('field_max', field_max)

        vocab_sizes = (
            field_max - field_min + 1
        ).tolist()
        self.field_embeddings = nn.ModuleList(
            nn.Embedding(int(vocab_size), embedding_dim)
            for vocab_size in vocab_sizes
        )
        self.field_identifiers = nn.Parameter(
            torch.empty(self.field_count, embedding_dim)
        )
        nn.init.normal_(self.field_identifiers, std=0.02)

        # 只使用目标字段前后帧、同帧其他字段和字段身份预测Cover期望。
        self.cover_context_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.residual_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalResidualBlock(hidden_dim, 2 ** layer)
            for layer in range(num_layers)
        )
        self.local_evidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 三种top-k比例负责稀疏证据；均值和方差负责弥散证据。
        self.pool_stat_dim = 7
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + self.pool_stat_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.classifier = nn.Linear(64, Num_class)

    @staticmethod
    def _temporal_neighbors(values):
        zero = torch.zeros_like(values[:, :1])
        previous = torch.cat((zero, values[:, :-1]), dim=1)
        following = torch.cat((values[:, 1:], zero), dim=1)
        return previous, following

    @staticmethod
    def _top_fraction_mean(values, fraction):
        count = max(1, int(round(values.size(1) * fraction)))
        return values.topk(count, dim=1).values.mean(dim=1)

    @staticmethod
    def _is_legacy_triplicate(x):
        """兼容模板测试中的三倍复制，但不把它当成训练设计。"""
        if x.size(0) % 3 != 0:
            return False
        return (
            torch.equal(x[0::3], x[1::3])
            and torch.equal(x[0::3], x[2::3])
        )

    def _embed_fields(self, x):
        tokens = []
        for local_id, embedding in enumerate(self.field_embeddings):
            values = x[:, :, local_id].long()
            minimum = self.field_min[local_id]
            maximum = self.field_max[local_id]
            if torch.any(values < minimum) or torch.any(values > maximum):
                original_id = self.feature_columns[local_id]
                raise ValueError(
                    f"字段{original_id}码字超出"
                    f"[{int(minimum)}, {int(maximum)}]"
                )
            tokens.append(embedding(values - minimum))
        return torch.stack(tokens, dim=2)

    def forward(self, x):
        if not self.training and self._is_legacy_triplicate(x):
            x = x[0::3]
        token = self._embed_fields(x)  # [B,T,C,E]
        previous, following = self._temporal_neighbors(token)
        token_sum = token.sum(dim=2, keepdim=True)
        if self.field_count > 1:
            other_current = (
                token_sum - token
            ) / float(self.field_count - 1)
        else:
            other_current = torch.zeros_like(token)
        field_id = self.field_identifiers.view(
            1, 1, self.field_count, -1
        ).expand(x.size(0), x.size(1), -1, -1)

        context = torch.cat(
            (previous, following, other_current, field_id), dim=-1
        )
        expected_token = self.cover_context_predictor(context)
        native_error = (
            expected_token - token.detach()
        ).pow(2).mean(dim=-1)

        # 分类梯度不更新Cover预测器；它只由Cover native loss训练。
        expected_for_residual = expected_token.detach()
        residual = token - expected_for_residual
        field_feature = self.residual_encoder(
            torch.cat(
                (residual, residual.abs(), token), dim=-1
            )
        )

        field_mean = field_feature.mean(dim=2)
        field_std = field_feature.std(dim=2, unbiased=False)
        field_max = field_feature.amax(dim=2)
        frame = self.frame_encoder(
            torch.cat((field_mean, field_std, field_max), dim=-1)
        )
        temporal = frame.transpose(1, 2)
        for block in self.temporal_blocks:
            temporal = block(temporal)
        temporal = temporal.transpose(1, 2)

        local_hidden = F.gelu(
            field_feature + temporal.unsqueeze(2)
        )
        local_evidence = self.local_evidence_head(
            local_hidden
        ).squeeze(-1)
        frame_evidence = torch.logsumexp(local_evidence, dim=2)
        frame_evidence = frame_evidence - local_evidence.new_tensor(
            float(self.field_count)
        ).log()

        attention = torch.softmax(frame_evidence / 0.70, dim=1)
        weighted_pool = torch.sum(
            attention.unsqueeze(-1) * temporal, dim=1
        )
        temporal_mean = temporal.mean(dim=1)
        temporal_std = temporal.std(dim=1, unbiased=False)

        pool_stats = torch.stack(
            (
                frame_evidence.mean(dim=1),
                frame_evidence.std(dim=1, unbiased=False),
                frame_evidence.amax(dim=1),
                frame_evidence.amin(dim=1),
                self._top_fraction_mean(frame_evidence, 0.10),
                self._top_fraction_mean(frame_evidence, 0.25),
                self._top_fraction_mean(frame_evidence, 0.50),
            ),
            dim=1,
        )
        features = self.feature_head(
            torch.cat(
                (
                    weighted_pool,
                    temporal_mean,
                    temporal_std,
                    pool_stats,
                ),
                dim=1,
            )
        )
        logits = self.classifier(features)
        aux = {
            'native_error': native_error,
            'local_evidence': local_evidence,
            'frame_evidence': frame_evidence,
        }
        return aux, logits, logits, features


class Classifier_CL(MSCRENet):
    def __init__(self, num_layers=Num_layers):
        super().__init__(
            num_layers=num_layers,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
            feature_columns=FEATURE_COLUMNS,
        )




def seed_everything(seed=TRAIN_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_embedding_rate(rate):
    """同时兼容CSV中的0.1--0.4和10--40两种写法。"""
    return torch.where(rate > 1.0, rate / 100.0, rate)


class MSCRELoss(nn.Module):
    """难样本聚焦 + Cover参考 + 嵌入强度有序约束。"""

    def __init__(
        self,
        gamma=FOCAL_GAMMA,
        native_weight=NATIVE_WEIGHT,
        order_weight=ORDER_WEIGHT,
        order_margin=ORDER_MARGIN,
    ):
        super().__init__()
        self.gamma = gamma
        self.native_weight = native_weight
        self.order_weight = order_weight
        self.order_margin = order_margin

    def _soft_focal_loss(self, logits, positive_probability):
        positive_probability = positive_probability.clamp(0.0, 1.0)
        target_distribution = torch.stack(
            (1.0 - positive_probability, positive_probability), dim=1
        )
        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        focal_factor = (1.0 - probability).pow(self.gamma)
        return -(
            target_distribution * focal_factor * log_probability
        ).sum(dim=1).mean()

    def _group_ordinal_loss(self, logits, labels):
        """约束组均值：Cover < 10% < 20% < 30% < 40%。"""
        binary_label = labels[:, 1].long()
        score = logits[:, 1] - logits[:, 0]
        ordered_scores = []

        cover_mask = binary_label == 0
        if cover_mask.any():
            ordered_scores.append(score[cover_mask].mean())

        rate = normalize_embedding_rate(labels[:, 0])
        positive_rate = torch.unique(rate[binary_label == 1]).sort().values
        for current_rate in positive_rate:
            mask = (binary_label == 1) & torch.isclose(
                rate,
                current_rate,
                atol=1e-5,
                rtol=0.0,
            )
            if mask.any():
                ordered_scores.append(score[mask].mean())

        if len(ordered_scores) < 2:
            return score.new_zeros(())
        penalties = [
            F.relu(left + self.order_margin - right)
            for left, right in zip(
                ordered_scores[:-1], ordered_scores[1:]
            )
        ]
        return torch.stack(penalties).mean()

    def forward(
        self,
        model_output,
        labels,
        positive_probability=None,
    ):
        auxiliary, logits, _, _ = model_output
        if positive_probability is None:
            positive_probability = labels[:, 1].float()

        classification_loss = self._soft_focal_loss(
            logits,
            positive_probability,
        )

        cover_mask = labels[:, 1] < 0.5
        if cover_mask.any():
            native_loss = auxiliary['native_error'][cover_mask].mean()
        else:
            native_loss = logits.new_zeros(())

        ordinal_loss = self._group_ordinal_loss(logits, labels)
        total_loss = (
            classification_loss
            + self.native_weight * native_loss
            + self.order_weight * ordinal_loss
        )
        loss_items = {
            'classification': classification_loss.detach(),
            'native': native_loss.detach(),
            'ordinal': ordinal_loss.detach(),
        }
        return total_loss, loss_items


def random_temporal_view(inputs, labels):
    """随机短时视图；软标签表示裁剪段含修改的概率。"""
    hard_probability = labels[:, 1].float()
    if random.random() >= CROP_PROBABILITY:
        return inputs, hard_probability

    ratio = random.choice(CROP_RATIOS)
    crop_length = max(1, int(round(inputs.size(1) * ratio)))
    if crop_length >= inputs.size(1):
        return inputs, hard_probability

    start = random.randint(0, inputs.size(1) - crop_length)
    cropped_inputs = inputs[:, start:start + crop_length]

    rate = normalize_embedding_rate(labels[:, 0]).clamp(0.0, 1.0)
    contains_change = 1.0 - torch.pow(1.0 - rate, crop_length)
    soft_probability = torch.where(
        labels[:, 1] > 0.5,
        contains_change,
        torch.zeros_like(contains_change),
    )
    return cropped_inputs, soft_probability


def harmonic_mean(values):
    """对任何一组为0的召回率给出接近0的分数。"""
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        return 0.0
    values = values.clamp_min(1e-8)
    return float(values.numel() / torch.reciprocal(values).sum())


def evaluate_model(model, val_loader, device):
    """按验证集各嵌入率BA的等权算术平均选择checkpoint和阈值。

    每率BA = (Cover召回率 + 该率Stego召回率) / 2。
    只平均验证集中实际存在的Stego嵌入率，不单独保护10%。
    所有嵌入率共用一个验证集选出的阈值，test不参与选择。
    """
    model.eval()
    positive_scores = []
    binary_labels = []
    embedding_rates = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, logits, _, _ = model(inputs)
            positive_scores.append(
                torch.softmax(logits, dim=1)[:, 1].cpu()
            )
            binary_labels.append(labels[:, 1].long().cpu())
            embedding_rates.append(
                normalize_embedding_rate(labels[:, 0]).cpu()
            )

    positive_score = torch.cat(positive_scores)
    binary_label = torch.cat(binary_labels)
    embedding_rate = torch.cat(embedding_rates)
    cover_mask = binary_label == 0
    positive_rates = torch.unique(
        embedding_rate[binary_label == 1]
    ).sort().values
    if not cover_mask.any() or positive_rates.numel() == 0:
        raise ValueError('平均BA验证需要Cover和至少一种嵌入率的Stego')
    rate_masks = [
        (binary_label == 1) & torch.isclose(
            embedding_rate,
            rate,
            atol=1e-5,
            rtol=0.0,
        )
        for rate in positive_rates
    ]

    # 保留原阈值网格；只按验证集平均BA选一个全局阈值。
    thresholds = torch.linspace(0.01, 0.99, 197)
    predicted_positive = (
        positive_score.unsqueeze(0) >= thresholds.unsqueeze(1)
    )
    cover_recalls = (
        ~predicted_positive[:, cover_mask]
    ).float().mean(dim=1)
    rate_recalls = [
        predicted_positive[:, mask].float().mean(dim=1)
        for mask in rate_masks
    ]
    rate_ba_matrix = 0.5 * (
        cover_recalls.unsqueeze(1) + torch.stack(rate_recalls, dim=1)
    )
    mean_ba_scores = rate_ba_matrix.mean(dim=1)

    # 分数相同时选更接0.5的阈值，减少不必要的校准偏移。
    maximum_score = mean_ba_scores.max()
    candidate_indices = torch.nonzero(
        torch.isclose(
            mean_ba_scores,
            maximum_score,
            atol=1e-8,
            rtol=0.0,
        ),
        as_tuple=False,
    ).flatten()
    nearest = torch.argmin(
        (thresholds[candidate_indices] - 0.5).abs()
    )
    best_index = int(candidate_indices[nearest])
    threshold = float(thresholds[best_index])

    prediction = (positive_score >= threshold).long()
    overall_accuracy = (
        prediction == binary_label
    ).float().mean().item()
    cover_recall = (
        prediction[cover_mask] == 0
    ).float().mean().item()

    rate_balanced_accuracy = {}
    rate_steg_recall = {}
    selected_recalls = [cover_recall]
    for rate, rate_mask in zip(positive_rates, rate_masks):
        rate_key = round(float(rate.item()), 4)
        steg_recall = (
            prediction[rate_mask] == 1
        ).float().mean().item()
        selected_recalls.append(steg_recall)
        rate_steg_recall[rate_key] = steg_recall
        rate_balanced_accuracy[rate_key] = 0.5 * (
            cover_recall + steg_recall
        )

    robust_score = harmonic_mean(selected_recalls)
    lowest_rate_recall = (
        rate_steg_recall[min(rate_steg_recall)]
        if rate_steg_recall else 0.0
    )
    hard_score = harmonic_mean(
        [cover_recall, lowest_rate_recall]
    )
    return {
        'overall': overall_accuracy,
        'mean_ba': float(mean_ba_scores[best_index]),
        'cover_recall': cover_recall,
        'robust_score': robust_score,
        'hard_score': hard_score,
        'threshold': threshold,
        'by_rate': rate_balanced_accuracy,
        'steg_recall_by_rate': rate_steg_recall,
    }


def train_model(
    model,
    train_steg_loader,
    train_cover_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
):
    best_score = -1.0
    stale_checks = 0
    checkpoint_prefix = result_path + os.sep

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        step_count = 0

        # 两个loader由原框架分别shuffle；合并后再次随机打乱。
        for steg_batch, cover_batch in zip(
            train_steg_loader,
            train_cover_loader,
        ):
            steg_inputs, steg_labels = steg_batch
            cover_inputs, cover_labels = cover_batch
            inputs = torch.cat((steg_inputs, cover_inputs), dim=0)
            labels = torch.cat((steg_labels, cover_labels), dim=0)
            permutation = torch.randperm(inputs.size(0))
            inputs = inputs[permutation].to(device, non_blocking=True)
            labels = labels[permutation].to(device, non_blocking=True)
            inputs, positive_probability = random_temporal_view(
                inputs,
                labels,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == 'cuda',
                dtype=torch.bfloat16,
            ):
                model_output = model(inputs)
                loss, _ = criterion(
                    model_output,
                    labels,
                    positive_probability,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

        mean_loss = running_loss / max(step_count, 1)
        if epoch % VALIDATE_EVERY != 0 and epoch != TRAIN_EPOCHS:
            print(
                f"epoch {epoch:03d}/{TRAIN_EPOCHS}, "
                f"loss {mean_loss:.6f}"
            )
            continue

        metrics = evaluate_model(model, val_loader, device)
        scheduler.step(metrics['mean_ba'])
        is_best = metrics['mean_ba'] > best_score
        if is_best:
            best_score = metrics['mean_ba']
            stale_checks = 0
        else:
            stale_checks += 1

        save_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'score': metrics['mean_ba'],
                'selection_metric': (
                    'mean_of_per_rate_balanced_accuracy'
                ),
                'threshold': metrics['threshold'],
                'validation_metrics': metrics,
                'feature_columns': FEATURE_COLUMNS,
            },
            is_best,
            checkpoint_prefix,
        )
        print(
            f"epoch {epoch:03d}/{TRAIN_EPOCHS}, "
            f"loss {mean_loss:.6f}, "
            f"val_acc {metrics['overall']:.4f}, "
            f"mean_ba {metrics['mean_ba']:.4f}"
        )

        if stale_checks >= EARLY_STOP_PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    return best_score


def save_checkpoint(state, is_best, prefix):
    if is_best:
        directory = os.path.dirname(prefix)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save model
        torch.save(state, prefix + 'model_best.pth.tar')
        print('save the best checkpoint :' + prefix + 'model_best.pth.tar')


def parse_sample_test(file_path):

    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:

        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample


def parse_test_args():
    parser = argparse.ArgumentParser(
        description='Train mixed-rate model or test an existing checkpoint.'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Load checkpoint and test only; skip training.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path used with --test-only.',
    )
    parser.add_argument(
        '--test-steg',
        type=str,
        default=None,
        help='Steg test directory used with --test-only.',
    )
    parser.add_argument(
        '--test-cover',
        type=str,
        default=None,
        help='Cover test directory used with --test-only.',
    )
    parser.add_argument(
        '--test-rate',
        type=int,
        default=10,
        help='Embedding rate label written to result.txt.',
    )
    return parser.parse_args()


def load_evaluation_checkpoint(checkpoint_path: str, target_device):
    """加载本文件第一阶段生成的基础checkpoint。"""
    checkpoint = torch.load(
        checkpoint_path, map_location=target_device, weights_only=True,
    )
    if tuple(checkpoint.get('feature_columns', FEATURE_COLUMNS)) != tuple(FEATURE_COLUMNS):
        raise ValueError('checkpoint的FEATURE_COLUMNS与当前代码不一致')
    if 'threshold' not in checkpoint:
        raise ValueError('checkpoint缺少threshold；请使用本次原训练保存的完整权重，不能猜测旧阈值')
    threshold = float(checkpoint['threshold'])
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError('checkpoint的threshold必须在(0,1)内')
    if checkpoint.get('model_type', 'base') != 'base':
        raise ValueError('此单阶段代码只接受基础模型checkpoint')
    if any(name.startswith('dense_') for name in checkpoint['model']):
        raise ValueError('这是其他v2/v4的dense修正权重，不属于本文件；请使用本文件训练生成的checkpoint')
    model = Classifier_CL(num_layers=Num_layers).to(target_device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint



def test_model_with_best_checkpoint(
    File_Embed,
    File_NoEmbed,
    emd_rate,
    checkpoint_path=None,
):

    if checkpoint_path is None:
        checkpoint_path = os.path.join(result_path, 'model_best.pth.tar')
    model, best_checkpoint = load_evaluation_checkpoint(checkpoint_path, device)
    print('load bestcheck from :', checkpoint_path)
    threshold = float(best_checkpoint.get('threshold', 0.5))
    validation_metrics = best_checkpoint.get('validation_metrics', {})
    print(
        'checkpoint epoch =', best_checkpoint.get('epoch'),
        'val_acc =', validation_metrics.get('overall'),
        'mean_ba =', validation_metrics.get('mean_ba'),
        'threshold =', threshold,
    )

    x_test, y_test = get_alter_loaders_test(File_Embed, File_NoEmbed)
    x_test = x_test[:, :, list(FEATURE_COLUMNS)]

    test_loader = convert_to_loader_test(x_test, y_test, BATCH_SIZE)

    model.eval()
    correct_preds = 0
    total_preds = 0
    cover_correct = 0
    cover_total = 0
    steg_correct = 0
    steg_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1), inputs.size(2)).to(device)
            for i in range(inputs.size(0)):
                input_final[3 * i] = inputs[i]
                input_final[3 * i + 1] = inputs[i]
                input_final[3 * i + 2] = inputs[i]
            _, outputs_sup_1, outputs_sup_2, _ = model(input_final)
            predicted_1 = (
                torch.softmax(outputs_sup_1, dim=1)[:, 1] >= threshold
            ).long()
            predicted_2 = (
                torch.softmax(outputs_sup_2, dim=1)[:, 1] >= threshold
            ).long()
            _, labels = torch.max(labels, 1)
            for prediction in (predicted_1, predicted_2):
                total_preds += labels.size(0)
                correct_preds += (prediction == labels).sum().item()
                cover_mask = labels == 0
                steg_mask = labels == 1
                cover_total += cover_mask.sum().item()
                steg_total += steg_mask.sum().item()
                cover_correct += (
                    prediction[cover_mask] == 0
                ).sum().item()
                steg_correct += (
                    prediction[steg_mask] == 1
                ).sum().item()


    accuracy = correct_preds / total_preds
    cover_recall = cover_correct / max(cover_total, 1)
    steg_recall = steg_correct / max(steg_total, 1)
    balanced_accuracy = 0.5 * (cover_recall + steg_recall)
    print(
        f"test Accuracy: {accuracy:.4f}, "
        f"balanced: {balanced_accuracy:.4f}, "
        f"cover recall: {cover_recall:.4f}, "
        f"steg recall: {steg_recall:.4f}, "
        f"threshold: {threshold:.3f}"
    )
    f = open(os.path.join(result_path, "result.txt"), 'a')
    f.write(
        "test %d%%, acc %.4f, balanced %.4f, cover_recall %.4f, "
        "steg_recall %.4f, threshold %.3f\n"
        % (
            emd_rate,
            accuracy,
            balanced_accuracy,
            cover_recall,
            steg_recall,
            threshold,
        )
    )
    f.close()


def convert_to_loader_test(x_test, y_test, batch_size):
    # 将numpy数组转换为PyTorch的Tensor
    x_test_tensor = torch.Tensor(x_test)
    y_test_tensor = torch.Tensor(y_test)

    # 创建训练集和测试集的数据集对象
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # 创建训练集和测试集的数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def get_alter_loaders_test(File_Embed, File_NoEmbed):
    # 定义文件夹列表
    FOLDERS = [
        {"class": 1, "folder": File_Embed},  # 正样本数据文件所在的文件夹
        {"class": 0, "folder": File_NoEmbed}  # 负样本数据文件所在的文件夹
    ]

    # 获取所有文件路径
    all_files = [(item, folder["class"]) for folder in FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(all_files)

    # 解析每个样本文件，并存储在列表中
    all_samples_x = [(parse_sample_test(item[0])) for item in all_files]
    all_samples_y = [item[1] for item in all_files]  # 获取样本标签
    np_all_samples_x = np.asarray(all_samples_x)  # 转换样本列表为numpy数组
    np_all_samples_y = np.asarray(all_samples_y)  # 转换标签列表为numpy数组

    # 使用OneHotEncoder进行标签的独热编码
    encoder = OneHotEncoder(categories='auto', sparse_output=False)

    # 划分训练集和测试集
    x_test = np_all_samples_x
    y_test_ori = np_all_samples_y

    # 对测试集标签进行独热编码
    y_test = encoder.fit_transform(y_test_ori.reshape(-1, 1))

    return x_test, y_test


def test():
    test_model_with_best_checkpoint(
        '/root/autodl-tmp/all_data/Quantified/Voip_retest/data/model_test/er/CNV-QIM_0.1/Steg',
        '/root/autodl-tmp/all_data/Quantified/Voip_retest/data/model_test/er/CNV-QIM_0.1/Cover', 10) # test file paths


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_test_args()
    if args.test_only:
        if args.checkpoint is None:
            raise ValueError('--test-only必须指定--checkpoint')
        if args.test_steg is None or args.test_cover is None:
            raise ValueError(
                '--test-only必须指定--test-steg和--test-cover'
            )
        test_model_with_best_checkpoint(
            args.test_steg,
            args.test_cover,
            args.test_rate,
            checkpoint_path=args.checkpoint,
        )
        raise SystemExit

    x_train, y_train, x_val, y_val = get_alter_loaders()

    # 这里只选择码字字段，不改变原数据划分、标签或测试框架。
    x_train = x_train[:, :, list(FEATURE_COLUMNS)]
    x_val = x_val[:, :, list(FEATURE_COLUMNS)]

    observed_rates = y_train[y_train[:, 1] == 1, 0].astype(float)
    observed_rates = np.where(
        observed_rates > 1.0,
        observed_rates / 100.0,
        observed_rates,
    )
    observed_rates = sorted(set(np.round(observed_rates, 4).tolist()))
    expected_rates = {0.1, 0.2, 0.3, 0.4}
    missing_rates = expected_rates.difference(observed_rates)
    if missing_rates:
        raise ValueError(
            "正式训练集应包含10%、20%、30%、40%嵌入率；"
            f"当前缺少{sorted(missing_rates)}。请检查训练路径和旧pkl。"
        )
    train_steg_loader, train_cover_loader, val_loader = (
        convert_to_loader_CL(
            x_train,
            y_train,
            x_val,
            y_val,
            BATCH_SIZE,
        )
    )
    model = Classifier_CL(num_layers=Num_layers).to(device)
    criterion = MSCRELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    base_best_score = train_model(
        model,
        train_steg_loader,
        train_cover_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
    )
    test()
