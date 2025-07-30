import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc, cohen_kappa_score
from sklearn.manifold import TSNE
import seaborn as sns
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from contextlib import nullcontext

SAMPLE_RATE = 16000
WINDOW_SIZE = 2.5  # seconds
WINDOW_OVERLAP = 1.0  # seconds
MEL_BANDS = 128
WINDOW_LENGTH = 25  # ms
HOP_LENGTH = 10  # ms
EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

CLASS_WEIGHTS = {
    'neutral': 1.1,
    'calm': 9.8,
    'happy': 1.0,
    'sad': 1.0,
    'angry': 1.0,
    'fear': 1.0,
    'disgust': 1.0,
    'surprise': 2.9
}

def save_model(model, optimizer, epoch, val_f1_macro, val_metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1_macro': val_f1_macro,
        'val_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in val_metrics.items() if k != 'class_report'}
    }, save_path)



def load_model(model, optimizer, model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('val_f1_macro', 0)


def load_and_preprocess_audio(file_path, target_duration=WINDOW_SIZE):
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    target_db = -20
    current_db = 20 * np.log10(np.mean(np.abs(waveform)) + 1e-8)
    gain = 10 ** ((target_db - current_db) / 20)
    waveform = waveform * gain
    
    target_length = int(target_duration * SAMPLE_RATE)
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        start = (len(waveform) - target_length) // 2
        waveform = waveform[start:start + target_length]
    
    waveform = waveform.astype(np.float32)
    
    return waveform

def extract_features(waveform, delta=True):
    n_fft = 1024  
    
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=SAMPLE_RATE, 
        n_fft=n_fft, 
        hop_length=int(HOP_LENGTH * SAMPLE_RATE / 1000), 
        win_length=int(WINDOW_LENGTH * SAMPLE_RATE / 1000), 
        n_mels=MEL_BANDS
    )
    
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    if not delta:
        return np.expand_dims(log_mel_spec, 0).astype(np.float32)
    
    features = np.stack([
        log_mel_spec, 
        librosa.feature.delta(log_mel_spec),
        librosa.feature.delta(log_mel_spec, order=2)
    ], axis=0).astype(np.float32)
    
    return features

def time_shift(waveform, shift_factor=0.1):
    shift = int(len(waveform) * shift_factor)
    direction = np.random.choice([-1, 1])
    shift = direction * shift
    shifted = np.roll(waveform, shift)
    return shifted

def spec_augment(features, max_time_mask=0.05, max_freq_mask=0.08, n_time_masks=2, n_freq_masks=2):
    features = features.copy()
    time_axis = features.shape[2]
    freq_axis = features.shape[1]
    
    for _ in range(n_time_masks):
        mask_size = int(time_axis * max_time_mask)
        if mask_size > 0:
            start = np.random.randint(0, time_axis - mask_size)
            features[:, :, start:start + mask_size] = 0
    
    for _ in range(n_freq_masks):
        mask_size = int(freq_axis * max_freq_mask)
        if mask_size > 0:
            start = np.random.randint(0, freq_axis - mask_size)
            features[:, start:start + mask_size, :] = 0
    
    return features

def apply_augmentation(waveform, features, emotion, emotion_id):

    if np.random.random() < 0.7:
        features = spec_augment(features)

    if emotion in ['calm', 'surprise'] or emotion_id in [1, 7]:
        if np.random.random() < 0.5:
            waveform = time_shift(waveform)
            
        if emotion == 'calm' or emotion_id == 1:
            if np.random.random() < 0.8:
                noise_level = np.random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_level, len(waveform))
                waveform = waveform + noise
                waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    
    return waveform, features

class EmotionDataset(Dataset):
    def __init__(self, csv_path, augment=False, feature_cache_dir=None, delta=True, dtype=torch.float32):
        self.data = pd.read_csv(csv_path)
        self.augment = augment
        self.feature_cache_dir = feature_cache_dir
        self.delta = delta
        self.dtype = dtype
        
        if feature_cache_dir and not os.path.exists(feature_cache_dir):
            os.makedirs(feature_cache_dir)
            
        self.file_paths = self.data['file_path'].values
        self.emotion_ids = self.data['emotion_id'].values
        self.emotions = self.data['emotion'].values
        
        if feature_cache_dir:
            self.cache_paths = [os.path.join(feature_cache_dir, 
                                 os.path.splitext(os.path.basename(fp))[0] + ".npy") 
                                 for fp in self.file_paths]
            self.cache_exists = [os.path.exists(cp) for cp in self.cache_paths]
        else:
            self.cache_paths = [None] * len(self.file_paths)
            self.cache_exists = [False] * len(self.file_paths)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        emotion_id = self.emotion_ids[idx]
        emotion = self.emotions[idx]

        if self.cache_exists[idx] and not self.augment:
            try:
                features = np.load(self.cache_paths[idx], mmap_mode='r')
                return torch.tensor(features, dtype=self.dtype), emotion_id
            except Exception as e:
                print(f"Cache loading failed for {file_path}: {e}")
                self.cache_exists[idx] = False
        
        waveform = load_and_preprocess_audio(file_path)
        features = extract_features(waveform, delta=self.delta)
        
        if self.augment:
            waveform, features = apply_augmentation(waveform, features, emotion, emotion_id)
        elif self.cache_paths[idx]:  
            try:
                np.save(self.cache_paths[idx], features)
                self.cache_exists[idx] = True
            except Exception as e:
                print(f"Failed to cache features for {file_path}: {e}")
        
        return torch.tensor(features, dtype=self.dtype), emotion_id

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.out_proj = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, input_dim)
        
        output = self.out_proj(context)
        
        return output, attn_weights


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(EmotionRecognitionModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.resblock1 = ResNetBlock(32, 32)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.resblock2 = ResNetBlock(64, 64)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.resblock3 = ResNetBlock(128, 128)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.resblock4 = ResNetBlock(256, 256)

        self.projection = None
        self.gru_input_dim = 256
        
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        
        self.attention = AttentionBlock(512, num_heads=4)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # i/p shape: [B, C, F, T] 

        if x.dtype != torch.float32 and not torch.is_autocast_enabled():
            x = x.to(torch.float32)
        x_dtype = x.dtype
    
        x = self.layer1(x)
        x = self.resblock1(x)
        
        x = self.layer2(x)
        x = self.resblock2(x)
        
        x = self.layer3(x)
        x = self.resblock3(x)
        
        x = self.layer4(x)
        x = self.resblock4(x)
        
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        x = x.reshape(batch_size, time, channels * freq)  # [B, T, C*F]

        input_dim = channels * freq
        if self.projection is None or self.projection.in_features != input_dim:
            self.projection = nn.Linear(input_dim, self.gru_input_dim).to(x.device).to(x_dtype)

        x = self.projection(x)  # [B, T, 256]
    
        gru_out, _ = self.gru(x)  # [B, T, 2*H]
        
        attn_out, attn_weights = self.attention(gru_out)
        
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        output = self.classifier(pooled)
        
        return output, attn_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()

        weights = torch.zeros(len(class_weights))
        for emotion, weight in class_weights.items():
            idx = EMOTION_CLASSES.index(emotion)
            weights[idx] = weight
        
        self.ce = nn.CrossEntropyLoss(weight=weights, reduction=reduction)
    
    def forward(self, inputs, targets):
        return self.ce(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.7, label_smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.wce = WeightedCrossEntropyLoss(class_weights)
        self.focal = FocalLoss(gamma=2.5)
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            inputs_log_softmax = F.log_softmax(inputs, dim=1)
            wce_loss = -torch.sum(targets_smooth * inputs_log_softmax, dim=1).mean()
        else:
            wce_loss = self.wce(inputs, targets)
        
        focal_loss = self.focal(inputs, targets)
        
        return self.alpha * wce_loss + (1 - self.alpha) * focal_loss

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    running_loss = 0.0
    
    batch_count = len(dataloader)
    all_preds = np.zeros(batch_count * dataloader.batch_size, dtype=np.int64)
    all_labels = np.zeros(batch_count * dataloader.batch_size, dtype=np.int64)
    
    idx = 0
    for features, labels in tqdm(dataloader, desc="Training"):
        try:
            features = features.to(device, non_blocking=True)  
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs, _ = model(features)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            
            batch_size = labels.size(0)
            _, preds = torch.max(outputs, 1)
            
            start_idx = idx * dataloader.batch_size
            end_idx = start_idx + batch_size
            all_preds[start_idx:end_idx] = preds.cpu().numpy()
            all_labels[start_idx:end_idx] = labels.cpu().numpy()
            
            idx += 1
            
            del features, labels, outputs, loss, preds
            
        except Exception as e:
            print(f"Error in training batch: {str(e)}")
            raise
    
    all_preds = all_preds[:end_idx]
    all_labels = all_labels[:end_idx]
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, accuracy, macro_f1, weighted_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_preds = np.zeros(len(dataloader.dataset), dtype=np.int64)
    all_labels = np.zeros(len(dataloader.dataset), dtype=np.int64)
    
    idx = 0
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            batch_size = labels.size(0)
            
            try:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs, _ = model(features)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                start_idx = idx * dataloader.batch_size
                end_idx = min(start_idx + batch_size, len(all_labels))
                
                all_probs.append(probs.cpu().numpy())
                all_preds[start_idx:end_idx] = preds.cpu().numpy()
                all_labels[start_idx:end_idx] = labels.cpu().numpy()
                
                idx += 1

                del features, labels, outputs, loss, probs, preds
                
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                raise
    
    all_preds = all_preds[:end_idx]
    all_labels = all_labels[:end_idx]
    all_probs = np.concatenate(all_probs, axis=0)[:end_idx]
    
    val_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    class_report = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        class_report[emotion] = {}
        
        y_true_binary = np.array(all_labels) == i
        y_pred_probs = all_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs)
        class_report[emotion]['precision'] = precision
        class_report[emotion]['recall'] = recall
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs)
        class_report[emotion]['fpr'] = fpr
        class_report[emotion]['tpr'] = tpr
        class_report[emotion]['auc'] = auc(fpr, tpr)
    
    metrics = {
        'loss': val_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': conf_matrix,
        'kappa': kappa,
        'class_report': class_report
    }
    
    return metrics

def process_long_audio(model, file_path, device):
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    window_samples = int(WINDOW_SIZE * sr)
    hop_samples = int((WINDOW_SIZE - WINDOW_OVERLAP) * sr)

    segments = []
    segment_times = []
    
    for start in range(0, len(waveform) - window_samples + 1, hop_samples):
        segment = waveform[start:start + window_samples]
        segments.append(segment)
        segment_times.append(start / sr)
    
    if not segments:
        return None, None
    
    max_batch_size = 32  
    predictions = []
    confidences = []
    attention_maps = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(segments), max_batch_size):
            batch_segments = segments[i:i+max_batch_size]

            batch_features = np.stack([extract_features(s) for s in batch_segments])
            batch_features = torch.tensor(batch_features, dtype=torch.float32).to(device)
            
            outputs, attention = model(batch_features)
            probs = F.softmax(outputs, dim=1)

            pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
            batch_confidences = torch.max(probs, dim=1)[0].cpu().numpy()
            
            predictions.extend(pred_indices.tolist())
            confidences.extend(batch_confidences.tolist())
            attention_maps.extend(attention.cpu().numpy())
    
    weighted_votes = np.zeros(NUM_CLASSES)
    for pred, conf in zip(predictions, confidences):
        weighted_votes[pred] += conf
    
    final_prediction = np.argmax(weighted_votes)

    trajectory = {
        'times': segment_times,
        'emotions': [EMOTION_CLASSES[p] for p in predictions],
        'confidences': confidences,
        'attention_maps': attention_maps
    }
    
    if len(predictions) > 2:
        smoothed_predictions = []
        for i in range(len(predictions)):
            if i == 0:
                smoothed_predictions.append(predictions[i])
            elif i == len(predictions) - 1:
                smoothed_predictions.append(predictions[i])
            else:
                window = predictions[i-1:i+2]
                counts = np.bincount(window, minlength=NUM_CLASSES)
                smoothed_predictions.append(np.argmax(counts))
                
        trajectory['smoothed_emotions'] = [EMOTION_CLASSES[p] for p in smoothed_predictions]
    else:
        trajectory['smoothed_emotions'] = trajectory['emotions']
    
    return EMOTION_CLASSES[final_prediction], trajectory


def create_visualizations(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for emotion in EMOTION_CLASSES:
        report = metrics['class_report'][emotion]
        plt.plot(report['fpr'], report['tpr'], label=f"{emotion} (AUC = {report['auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for emotion in ['surprise', 'calm']:
        report = metrics['class_report'][emotion]
        plt.plot(report['recall'], report['precision'], label=emotion)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Minority Classes')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close()


def visualize_emotion_trajectory(trajectory, audio_path, save_path):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.title('Emotion Trajectory')
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    emotion_colors = {emotion: colors[i] for i, emotion in enumerate(EMOTION_CLASSES)}
    
    for i, (time, emotion, conf) in enumerate(zip(trajectory['times'], trajectory['emotions'], trajectory['confidences'])):
        plt.scatter(time, EMOTION_CLASSES.index(emotion), color=emotion_colors[emotion], alpha=conf)
    
    plt.yticks(range(NUM_CLASSES), EMOTION_CLASSES)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.title('Smoothed Emotion Trajectory')
    
    for i, (time, emotion) in enumerate(zip(trajectory['times'], trajectory['smoothed_emotions'])):
        plt.scatter(time, EMOTION_CLASSES.index(emotion), color=emotion_colors[emotion])
        if i > 0:
            prev_time = trajectory['times'][i-1]
            prev_emotion = trajectory['smoothed_emotions'][i-1]
            plt.plot([prev_time, time], 
                     [EMOTION_CLASSES.index(prev_emotion), EMOTION_CLASSES.index(emotion)],
                     color='gray', alpha=0.5)
    
    plt.yticks(range(NUM_CLASSES), EMOTION_CLASSES)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.title('Prediction Confidence')
    plt.plot(trajectory['times'], trajectory['confidences'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Confidence')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(train_csv, val_csv, test_csv, output_dir, feature_cache_dir=None, batch_size=64, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dtype = torch.float32
    
    train_dataset = EmotionDataset(train_csv, augment=True, feature_cache_dir=feature_cache_dir, dtype=dtype)
    val_dataset = EmotionDataset(val_csv, augment=False, feature_cache_dir=feature_cache_dir, dtype=dtype)
    test_dataset = EmotionDataset(test_csv, augment=False, feature_cache_dir=feature_cache_dir, dtype=dtype)
    
    train_df = pd.read_csv(train_csv)
    class_counts = train_df['emotion_id'].value_counts().sort_index().values
    num_samples = len(train_df)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights) * num_samples
    
    sample_weights = [class_weights[label] for label in train_df['emotion_id']]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=min(4, os.cpu_count()), 
        pin_memory=True,
        persistent_workers=True,  
        prefetch_factor=2  
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = EmotionRecognitionModel(input_channels=3 if train_dataset.delta else 1)
    
    model = model.to(dtype).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    
    criterion = CombinedLoss(CLASS_WEIGHTS, alpha=0.7, label_smoothing=0.1)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=10000
    )
    
    scaler = GradScaler('cuda')
    
    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    os.makedirs(output_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1_macro': [],
        'train_f1_weighted': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'lr': []
    }
    
    early_stop_counter = 0
    early_stop_patience = 10
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        train_loss, train_acc, train_f1_macro, train_f1_weighted = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler
        )

        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1_macro = val_metrics['macro_f1']
        val_f1_weighted = val_metrics['weighted_f1']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1_macro'].append(train_f1_macro)
        history['train_f1_weighted'].append(train_f1_weighted)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1 Macro: {train_f1_macro:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1 Macro: {val_f1_macro:.4f}")
        
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            save_model(model, optimizer, epoch, val_f1_macro, val_metrics, best_model_path)
            print(f"Model saved with F1 Macro: {val_f1_macro:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1_macro'], label='Train')
    plt.plot(history['val_f1_macro'], label='Validation')
    plt.title('F1 Macro')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    _, _ = load_model(model, None, best_model_path, device)

    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {test_metrics['macro_f1']:.4f}")
    print(f"F1 Weighted: {test_metrics['weighted_f1']:.4f}")
    print(f"Cohen's Kappa: {test_metrics['kappa']:.4f}")
    
    create_visualizations(test_metrics, os.path.join(output_dir, 'visualizations'))
    
    with open(os.path.join(output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"F1 Macro: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"F1 Weighted: {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Cohen's Kappa: {test_metrics['kappa']:.4f}\n")
        
        f.write("\nPer-class metrics:\n")
        conf_matrix = test_metrics['confusion_matrix']
        class_precision = conf_matrix.diagonal() / conf_matrix.sum(axis=0)
        class_recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
        
        for i, emotion in enumerate(EMOTION_CLASSES):
            f.write(f"{emotion}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1-score: {class_f1[i]:.4f}\n")
            f.write(f"  AUC-ROC: {test_metrics['class_report'][emotion]['auc']:.4f}\n")
    
    return model, test_metrics




def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    def get_embeddings(model, x):
        
        x = model.layer1(x)
        x = model.resblock1(x)
        
        x = model.layer2(x)
        x = model.resblock2(x)
        
        x = model.layer3(x)
        x = model.resblock3(x)
        
        x = model.layer4(x)
        x = model.resblock4(x)

        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time, channels * freq)

        if model.projection is not None:
            x = model.projection(x)
        
        gru_out, _ = model.gru(x)
        
        attn_out, _ = model.attention(gru_out)
        
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        
        
        embeddings = torch.cat([avg_pool, max_pool], dim=1)
        
        return embeddings
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Extracting embeddings"):
            if features.dtype != torch.float32 and not torch.is_autocast_enabled():
                features = features.to(torch.float32)
                
            features = features.to(device)
            
            embeddings = get_embeddings(model, features)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0), np.array(all_labels)

def visualize_tsne(model, dataloader, device, save_path):
    embeddings, labels = extract_embeddings(model, dataloader, device)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    
    for i, emotion in enumerate(EMOTION_CLASSES):
        idx = labels == i
        plt.scatter(embeddings_tsne[idx, 0], embeddings_tsne[idx, 1], 
                    color=colors[i], label=emotion, alpha=0.7)
    
    plt.title('t-SNE Visualization of Emotion Embeddings')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main():
    train_csv = 'dataset/train.csv'
    val_csv = 'dataset/val.csv'
    test_csv = 'dataset/test.csv'
    output_dir = 'results'
    feature_cache_dir = 'features_cache'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_cache_dir, exist_ok=True)
    
    model, test_metrics = train_model(
        train_csv, val_csv, test_csv,
        output_dir=output_dir,
        feature_cache_dir=feature_cache_dir,
        batch_size=32,
        epochs=50
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = EmotionDataset(test_csv, augment=False, feature_cache_dir=feature_cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    visualize_tsne(model, test_loader, device, os.path.join(output_dir, 'visualizations', 'tsne.png'))
    
    print("Running demo for long audio processing...")
    test_df = pd.read_csv(test_csv)
    sample_file = test_df.iloc[0]['file_path']
    
    prediction, trajectory = process_long_audio(model, sample_file, device)
    print(f"Predicted emotion: {prediction}")
    
    if trajectory:
        visualize_emotion_trajectory(
            trajectory, 
            sample_file, 
            os.path.join(output_dir, 'visualizations', 'emotion_trajectory.png')
        )

def predict_emotion(model_path, audio_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EmotionRecognitionModel(input_channels=3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    waveform = load_and_preprocess_audio(audio_path)
    features = extract_features(waveform)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        with autocast('cuda') if device.type == 'cuda' else nullcontext():
            outputs, attention = model(features_tensor)
            probs = F.softmax(outputs, dim=1)
            
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
    
    prediction = EMOTION_CLASSES[pred_idx]
    top3_idx = torch.argsort(probs[0], descending=True)[:3].cpu().numpy()
    top3_emotions = [(EMOTION_CLASSES[idx], probs[0, idx].item()) for idx in top3_idx]
    
    return prediction, confidence, top3_emotions, attention.cpu().numpy()

def batch_process_audio_files(model_path, file_list, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionRecognitionModel(input_channels=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    results = []

    for file_path in tqdm(file_list, desc="Processing files"):
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        duration = len(waveform) / sr
        
        if duration > 10:
            prediction, trajectory = process_long_audio(model, file_path, device)
            confidence = max(trajectory['confidences']) if trajectory else 0.0
        else:
            prediction, confidence, _, _ = predict_emotion(model_path, file_path)
        
        results.append({
            'file_path': file_path,
            'prediction': prediction,
            'confidence': confidence,
            'duration': duration
        })
    
    pd.DataFrame(results).to_csv(output_csv, index=False)
    
    return results


if __name__ == '__main__':
    main()