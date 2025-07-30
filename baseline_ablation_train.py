import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from scipy import stats
import warnings
from datetime import datetime
import gc
import hashlib
import traceback
import shutil
warnings.filterwarnings('ignore')

SAMPLE_RATE = 16000
WINDOW_SIZE = 2.5
MEL_BANDS = 128
WINDOW_LENGTH = 25
HOP_LENGTH = 10
EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

CLASS_WEIGHTS = {
    'neutral': 1.1, 'calm': 9.8, 'happy': 1.0, 'sad': 1.0,
    'angry': 1.0, 'fear': 1.0, 'disgust': 1.0, 'surprise': 2.9
}

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, 'experiment.log')
        self.log_file = open(self.log_file_path, 'w', buffering=1)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + '\n')
        self.log_file.flush()
        
    def close(self):
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.close()

def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")
    return True

def get_cache_key(file_path):
    validate_file_exists(file_path)
    file_stat = os.stat(file_path)
    content = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()

def load_and_preprocess_audio(file_path, target_duration=WINDOW_SIZE):
    validate_file_exists(file_path)
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    if len(waveform) == 0:
        raise ValueError(f"Empty audio file: {file_path}")
    
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
    
    return waveform.astype(np.float32)

def extract_deep_features(waveform, delta=True):
    if len(waveform) == 0:
        raise ValueError("Empty waveform provided")
    
    n_fft = 1024
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=n_fft,
        hop_length=int(HOP_LENGTH * SAMPLE_RATE / 1000),
        win_length=int(WINDOW_LENGTH * SAMPLE_RATE / 1000),
        n_mels=MEL_BANDS
    )
    
    if mel_spec.size == 0:
        raise ValueError("Failed to extract mel spectrogram")
    
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    if not delta:
        return np.expand_dims(log_mel_spec, 0).astype(np.float32)
    
    delta_1 = librosa.feature.delta(log_mel_spec)
    delta_2 = librosa.feature.delta(log_mel_spec, order=2)
    
    features = np.stack([log_mel_spec, delta_1, delta_2], axis=0).astype(np.float32)
    
    if features.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {features.shape[0]}")
    
    return features

def extract_handcrafted_features(waveform, sr=SAMPLE_RATE):
    if len(waveform) == 0:
        raise ValueError("Empty waveform provided")
    
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    rmse = librosa.feature.rms(y=waveform)
    spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
    
    all_features = [mfccs, zcr, rmse, spectral_centroids, spectral_rolloff, chroma, spectral_bandwidth]
    
    for i, feat in enumerate(all_features):
        if feat.size == 0:
            raise ValueError(f"Feature {i} extraction failed")
    
    features = []
    for feat in all_features:
        feat_stats = [np.mean(feat), np.std(feat), np.max(feat), np.min(feat), np.median(feat), np.var(feat)]
        for stat in feat_stats:
            if not np.isfinite(stat):
                raise ValueError("Non-finite feature value detected")
        features.extend(feat_stats)
    
    result = np.array(features, dtype=np.float32)
    
    if len(result) != 42:
        raise ValueError(f"Expected 42 features, got {len(result)}")
    
    return result

def preprocess_and_cache_features(csv_files, logger):
    logger.log("Starting comprehensive feature caching...")
    
    all_files = set()
    for csv_file in csv_files:
        validate_file_exists(csv_file)
        df = pd.read_csv(csv_file)
        if df.empty:
            raise ValueError(f"Empty CSV file: {csv_file}")
        if 'file_path' not in df.columns:
            raise ValueError(f"CSV file missing 'file_path' column: {csv_file}")
        all_files.update(df['file_path'].values)
    
    deep_cache_dir = 'feature_cache_deep'
    ml_cache_dir = 'feature_cache_ml'
    os.makedirs(deep_cache_dir, exist_ok=True)
    os.makedirs(ml_cache_dir, exist_ok=True)
    
    cache_info_file = 'feature_cache_info.json'
    if os.path.exists(cache_info_file):
        with open(cache_info_file, 'r') as f:
            cache_info = json.load(f)
    else:
        cache_info = {}
    
    files_to_process = []
    existing_files = []
    
    for file_path in all_files:
        if not os.path.exists(file_path):
            logger.log(f"Warning: File not found, skipping: {file_path}")
            continue
            
        cache_key = get_cache_key(file_path)
        deep_cache_file = os.path.join(deep_cache_dir, f"{cache_key}.npy")
        ml_cache_file = os.path.join(ml_cache_dir, f"{cache_key}.npy")
        
        cache_valid = (
            cache_key in cache_info and 
            os.path.exists(deep_cache_file) and 
            os.path.exists(ml_cache_file) and
            os.path.getsize(deep_cache_file) > 0 and
            os.path.getsize(ml_cache_file) > 0
        )
        
        if cache_valid:
            existing_files.append(file_path)
        else:
            files_to_process.append((file_path, cache_key, deep_cache_file, ml_cache_file))
    
    logger.log(f"Found {len(existing_files)} files with valid cache")
    logger.log(f"Need to process {len(files_to_process)} files")
    
    if files_to_process:
        logger.log("Processing files for feature caching...")
        
        processed_count = 0
        failed_count = 0
        
        for file_path, cache_key, deep_cache_file, ml_cache_file in tqdm(files_to_process, desc="Caching features"):
            try:
                waveform = load_and_preprocess_audio(file_path)
                
                if not os.path.exists(deep_cache_file):
                    deep_features = extract_deep_features(waveform, delta=True)
                    np.save(deep_cache_file, deep_features)
                    
                    if not os.path.exists(deep_cache_file):
                        raise IOError(f"Failed to save deep cache file: {deep_cache_file}")
                
                if not os.path.exists(ml_cache_file):
                    ml_features = extract_handcrafted_features(waveform)
                    np.save(ml_cache_file, ml_features)
                    
                    if not os.path.exists(ml_cache_file):
                        raise IOError(f"Failed to save ML cache file: {ml_cache_file}")
                
                cache_info[cache_key] = {
                    'file_path': file_path,
                    'deep_cache': deep_cache_file,
                    'ml_cache': ml_cache_file,
                    'timestamp': datetime.now().isoformat()
                }
                
                processed_count += 1
                
            except Exception as e:
                logger.log(f"Failed to process {file_path}: {str(e)}")
                failed_count += 1
                
                if os.path.exists(deep_cache_file):
                    os.remove(deep_cache_file)
                if os.path.exists(ml_cache_file):
                    os.remove(ml_cache_file)
                continue
        
        with open(cache_info_file, 'w') as f:
            json.dump(cache_info, f, indent=2)
        
        logger.log(f"Feature caching completed: {processed_count} processed, {failed_count} failed")
    else:
        logger.log("All features already cached")
    
    return cache_info_file

class CachedDataset(Dataset):
    def __init__(self, csv_path, feature_type='deep', dtype=torch.float32):
        validate_file_exists(csv_path)
        
        self.data = pd.read_csv(csv_path)
        if self.data.empty:
            raise ValueError(f"Empty CSV file: {csv_path}")
        
        required_columns = ['file_path', 'emotion_id', 'emotion', 'subdataset']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")
        
        self.feature_type = feature_type
        self.dtype = dtype
        
        cache_info_file = 'feature_cache_info.json'
        validate_file_exists(cache_info_file)
        
        with open(cache_info_file, 'r') as f:
            self.cache_info = json.load(f)
        
        if not self.cache_info:
            raise ValueError("Empty cache info file")
        
        self.valid_samples = []
        
        for idx, row in self.data.iterrows():
            file_path = row['file_path']
            
            if not os.path.exists(file_path):
                continue
            
            cache_key = get_cache_key(file_path)
            
            if cache_key not in self.cache_info:
                continue
            
            cache_data = self.cache_info[cache_key]
            cache_file = cache_data[f'{feature_type}_cache']
            
            if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
                continue
            
            self.valid_samples.append({
                'cache_key': cache_key,
                'cache_file': cache_file,
                'emotion_id': row['emotion_id'],
                'subdataset': row['subdataset']
            })
        
        if not self.valid_samples:
            raise ValueError(f"No valid cached samples found for {csv_path}")
        
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.valid_samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.valid_samples)}")
        
        sample = self.valid_samples[idx]
        cache_file = sample['cache_file']
        emotion_id = sample['emotion_id']
        subdataset = sample['subdataset']
        
        features = np.load(cache_file, mmap_mode='r')
        
        if features.size == 0:
            raise ValueError(f"Empty cached features: {cache_file}")
        
        if self.feature_type == 'deep':
            if len(features.shape) != 3 or features.shape[0] != 3:
                raise ValueError(f"Invalid deep features shape: {features.shape}")
            
            features_copy = features.copy()
            return torch.tensor(features_copy, dtype=self.dtype), emotion_id, subdataset
        else:
            if len(features.shape) != 1 or features.shape[0] != 42:
                raise ValueError(f"Invalid ML features shape: {features.shape}")
            
            features_copy = features.copy()
            return features_copy, emotion_id, subdataset

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

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionBlock, self).__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim {input_dim} must be divisible by num_heads {num_heads}")
        
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

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x, None

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.lstm = None
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, channels, freq, time = x.size()
        x = self.cnn(x)
        _, c_out, f_out, t_out = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, t_out, c_out * f_out)
        
        lstm_input_size = c_out * f_out
        if self.lstm is None:
            self.lstm = nn.LSTM(lstm_input_size, 64, batch_first=True, bidirectional=True).to(x.device)
        
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        output = self.classifier(pooled)
        return output, None

class NoAttentionResoNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(NoAttentionResoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock1 = ResNetBlock(32, 32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock2 = ResNetBlock(64, 64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock3 = ResNetBlock(128, 128)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock4 = ResNetBlock(256, 256)
        self.projection = None
        self.gru_input_dim = 256
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.float()  # Simplified dtype handling
        
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.layer2(x)
        x = self.resblock2(x)
        x = self.layer3(x)
        x = self.resblock3(x)
        x = self.layer4(x)
        x = self.resblock4(x)
        
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time, channels * freq)
        
        input_dim = channels * freq
        if self.projection is None:
            self.projection = nn.Linear(input_dim, self.gru_input_dim).to(x.device)
            self.add_module('dynamic_projection', self.projection)
        elif self.projection.in_features != input_dim:
            # Use global average pooling to fix dimension mismatch
            x = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, features]
            x = x.expand(batch_size, time, self.projection.in_features)  # Expand back
            
        x = self.projection(x)
        gru_out, _ = self.gru(x)
        avg_pool = torch.mean(gru_out, dim=1)
        output = self.classifier(avg_pool)
        return output, None

class NoGRUResoNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(NoGRUResoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock1 = ResNetBlock(32, 32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock2 = ResNetBlock(64, 64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock3 = ResNetBlock(128, 128)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.resblock4 = ResNetBlock(256, 256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.layer2(x)
        x = self.resblock2(x)
        x = self.layer3(x)
        x = self.resblock3(x)
        x = self.layer4(x)
        x = self.resblock4(x)
        x = self.classifier(x)
        return x, None

class NoResNetResoNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(NoResNetResoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)
        )
        self.projection = None
        self.gru_input_dim = 256
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.attention = AttentionBlock(512, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.float()  # Simplified dtype handling
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time, channels * freq)
        
        input_dim = channels * freq
        if self.projection is None:
            self.projection = nn.Linear(input_dim, self.gru_input_dim).to(x.device)
            self.add_module('dynamic_projection', self.projection)
        elif self.projection.in_features != input_dim:
            # Use global average pooling to fix dimension mismatch
            x = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, features]
            x = x.expand(batch_size, time, self.projection.in_features)  # Expand back
        
        x = self.projection(x)
        gru_out, _ = self.gru(x)
        attn_out, attn_weights = self.attention(gru_out)
        
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        output = self.classifier(pooled)
        return output, attn_weights

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        weights = torch.zeros(len(class_weights))
        for emotion, weight in class_weights.items():
            idx = EMOTION_CLASSES.index(emotion)
            weights[idx] = weight
        self.ce = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, inputs, targets):
        return self.ce(inputs, targets)

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler, logger):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # CRITICAL: Disable mixed precision for ResoNet models
    model_name = model.__class__.__name__
    use_mixed_precision = model_name in ['SimpleCNN', 'CNNLSTM']
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (features, labels, _) in enumerate(progress_bar):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Use mixed precision ONLY for simple models
        if use_mixed_precision:
            with autocast('cuda'):
                outputs, _ = model(features)
                loss = criterion(outputs, labels)
        else:
            # Full precision for complex models
            outputs, _ = model(features)
            loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.log(f"WARNING: Invalid loss detected: {loss.item()}, skipping batch {batch_idx}")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Different backprop for mixed vs full precision
        if use_mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # Even more conservative
            optimizer.step()
        
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / max(len([x for x in range(len(dataloader)) if x not in []]), 1)
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0.0
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0.0
    
    return epoch_loss, accuracy, macro_f1, weighted_f1

def validate(model, dataloader, criterion, device, logger):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_subdatasets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for features, labels, subdatasets in progress_bar:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast('cuda'):
                outputs, _ = model(features)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subdatasets.extend(subdatasets)
    
    val_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    per_dataset_acc = {}
    
    for dataset in set(all_subdatasets):
        dataset_mask = [i for i, d in enumerate(all_subdatasets) if d == dataset]
        if dataset_mask:
            dataset_preds = [all_preds[i] for i in dataset_mask]
            dataset_labels = [all_labels[i] for i in dataset_mask]
            per_dataset_acc[dataset] = accuracy_score(dataset_labels, dataset_preds)
    
    return {
        'loss': val_loss, 'accuracy': accuracy, 'macro_f1': macro_f1,
        'weighted_f1': weighted_f1, 'confusion_matrix': conf_matrix,
        'predictions': all_preds, 'labels': all_labels,
        'per_class_f1': per_class_f1, 'per_dataset_acc': per_dataset_acc
    }

def get_learning_rate(model_name, num_params):
    """Ultra-conservative learning rates for stability"""
    if 'ResoNet' in model_name:  
        return 5e-5  
    elif num_params > 500_000:  
        return 1e-4
    else:  
        return 2e-4




def train_model(model, train_loader, val_loader, test_loader, device, epochs, model_name, results_dir, logger):
    num_params = sum(p.numel() for p in model.parameters())
    lr = get_learning_rate(model_name, num_params)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = WeightedCrossEntropyLoss(CLASS_WEIGHTS).to(device)
    
    
    # SAFEST ALTERNATIVE: Use ReduceLROnPlateau to automatically lower LR on validation F1-score stagnation.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',          # Reduce LR when the metric stops increasing (correct for F1 score)
        factor=0.2,          # new_lr = lr * 0.2
        patience=3,          # Wait 3 epochs with no improvement before reducing LR
    )
    logger.log(f"Using ReduceLROnPlateau scheduler with initial lr={lr}")   
    
    scaler = GradScaler('cuda')

    best_f1 = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_weighted_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_weighted_f1': [], 'lr': []
    }
    
    logger.log(f"Training {model_name} for {epochs} epochs")
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        train_loss, train_acc, train_f1, train_wf1 = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler, logger
        )
        val_metrics = validate(model, val_loader, criterion, device, logger)
        scheduler.step(val_metrics['macro_f1'])
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_weighted_f1'].append(train_wf1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        history['lr'].append(scheduler.get_last_lr()[0])
        
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), f'{results_dir}/best_{model_name}.pth')
            logger.log(f"New best model saved with F1: {best_f1:.4f}")
        
        logger.log(f"Epoch {epoch+1}/{epochs} - Train: L={train_loss:.4f}, A={train_acc:.4f}, F1={train_f1:.4f} | Val: L={val_metrics['loss']:.4f}, A={val_metrics['accuracy']:.4f}, F1={val_metrics['macro_f1']:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    model.load_state_dict(torch.load(f'{results_dir}/best_{model_name}.pth'))
    test_metrics = validate(model, test_loader, criterion, device, logger)
    
    
    logger.log(f"{model_name} final test results: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['macro_f1']:.4f}")
    
    return test_metrics, history

def run_ml_baselines(train_dataset, val_dataset, test_dataset, results_dir, logger):
    logger.log("Extracting handcrafted features for ML baselines...")
    
    def extract_features_from_dataset(dataset, desc):
        X, y, datasets = [], [], []
        for features, label, subdataset in tqdm(dataset, desc=desc):
            if features.size == 0:
                raise ValueError(f"Empty features detected in {desc}")
            X.append(features)
            y.append(label)
            datasets.append(subdataset)
        return np.array(X), np.array(y), datasets
    
    X_train, y_train, _ = extract_features_from_dataset(train_dataset, "Train features")
    X_val, y_val, _ = extract_features_from_dataset(val_dataset, "Val features")
    X_test, y_test, test_datasets = extract_features_from_dataset(test_dataset, "Test features")
    
    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("No valid features extracted for ML baselines")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    with open(f'{results_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=15, min_samples_split=5),
        'SVM': SVC(random_state=42, probability=True, kernel='rbf', C=1.0, gamma='scale'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, min_samples_split=5)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.log(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        
        per_dataset_acc = {}
        for dataset in set(test_datasets):
            dataset_mask = [i for i, d in enumerate(test_datasets) if d == dataset]
            if dataset_mask:
                dataset_preds = [y_pred[i] for i in dataset_mask]
                dataset_labels = [y_test[i] for i in dataset_mask]
                per_dataset_acc[dataset] = accuracy_score(dataset_labels, dataset_preds)
        
        results[name] = {
            'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'confusion_matrix': conf_matrix.tolist(), 'predictions': y_pred.tolist(),
            'labels': y_test.tolist(), 'per_class_f1': per_class_f1.tolist(),
            'per_dataset_acc': per_dataset_acc, 'probabilities': y_proba.tolist() if y_proba is not None else None
        }
        
        logger.log(f"{name} - Accuracy: {accuracy:.4f}, F1 Macro: {macro_f1:.4f}, F1 Weighted: {weighted_f1:.4f}")
        
        with open(f'{results_dir}/ml_baseline_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    return results


import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_architecture_description(model_name):
    descriptions = {
        'DecisionTree': 'Decision Tree',
        'SVM': 'Support Vector Machine',
        'RandomForest': 'Random Forest',
        'SimpleCNN': 'CNN (3 layers)',
        'CNNLSTM': 'CNN + BiLSTM',
        'NoAttentionResoNet': 'CNN + ResNet + BiGRU',
        'NoGRUResoNet': 'CNN + ResNet',
        'NoResNetResoNet': 'CNN + BiGRU + Attention'
    }
    return descriptions[model_name] if model_name in descriptions else 'Unknown'

def get_parameter_count(model_name):
    param_counts = {
        'DecisionTree': 'N/A',
        'SVM': 'N/A', 
        'RandomForest': 'N/A',
        'SimpleCNN': '~50K',
        'CNNLSTM': '~150K',
        'NoAttentionResoNet': '~700K',
        'NoGRUResoNet': '~400K',
        'NoResNetResoNet': '~600K'
    }
    return param_counts[model_name] if model_name in param_counts else 'Unknown'

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_mode = input("Run in test mode? (1 epoch for each model) [y/N]: ").lower().strip() == 'y'
    epochs = 1 if test_mode else 50
    batch_size = 32 if test_mode else 64

    results_dir = f'baseline_results'
    os.makedirs(results_dir, exist_ok=True)
    
    logger = Logger(results_dir)
    
    validate_file_exists('train.csv')
    validate_file_exists('val.csv')
    validate_file_exists('test.csv')
    
    logger.log(f"Starting baseline experiment in {'test' if test_mode else 'full'} mode")
    logger.log(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}")
    
    all_results = {}
    histories = {}
    
    try:
        preprocess_and_cache_features(['train.csv', 'val.csv', 'test.csv'], logger)
        
        logger.log("Loading cached datasets...")
        train_dataset_deep = CachedDataset('train.csv', feature_type='deep')
        val_dataset_deep = CachedDataset('val.csv', feature_type='deep')
        test_dataset_deep = CachedDataset('test.csv', feature_type='deep')
        
        train_dataset_ml = CachedDataset('train.csv', feature_type='ml')
        val_dataset_ml = CachedDataset('val.csv', feature_type='ml')
        test_dataset_ml = CachedDataset('test.csv', feature_type='ml')
        
        logger.log(f"Dataset sizes - Train: {len(train_dataset_deep)}, Val: {len(val_dataset_deep)}, Test: {len(test_dataset_deep)}")
        
        train_loader = DataLoader(train_dataset_deep, batch_size=batch_size, shuffle=True, 
                                 num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
        val_loader = DataLoader(val_dataset_deep, batch_size=batch_size, shuffle=False, 
                               num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset_deep, batch_size=batch_size, shuffle=False, 
                                num_workers=2, pin_memory=True, persistent_workers=True)
        
        logger.log("Running ML baselines...")
        ml_results = run_ml_baselines(train_dataset_ml, val_dataset_ml, test_dataset_ml, results_dir, logger)
        all_results.update(ml_results)
        
        models = {
            'SimpleCNN': SimpleCNN(),
            'CNNLSTM': CNNLSTM(),
            'NoAttentionResoNet': NoAttentionResoNet(),
            'NoGRUResoNet': NoGRUResoNet(),
            'NoResNetResoNet': NoResNetResoNet()
        }
        
        for name, model in models.items():
            logger.log(f"Starting training for {name}")
            model = model.to(device)
            initialize_weights(model)
            
            test_metrics, history = train_model(
                model, train_loader, val_loader, test_loader, 
                device, epochs, name, results_dir, logger
            )
            
            all_results[name] = {
                'accuracy': test_metrics['accuracy'],
                'macro_f1': test_metrics['macro_f1'],
                'weighted_f1': test_metrics['weighted_f1'],
                'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
                'predictions': test_metrics['predictions'],
                'labels': test_metrics['labels'],
                'per_class_f1': test_metrics['per_class_f1'].tolist(),
                'per_dataset_acc': test_metrics['per_dataset_acc']
            }
            histories[name] = history
            
            del model
            torch.cuda.empty_cache()
            gc.collect()

        logger.log(f"\n{'='*50}")
        logger.log("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.log(f"{'='*50}")

        if all_results:
            best_model = max(all_results.items(), key=lambda x: x[1]['macro_f1'])
            logger.log(f"\nBest performing model: {best_model[0]}")
            logger.log(f"Accuracy: {best_model[1]['accuracy']:.4f}")
            logger.log(f"F1-Macro: {best_model[1]['macro_f1']:.4f}")
            logger.log(f"F1-Weighted: {best_model[1]['weighted_f1']:.4f}")
        
        if test_mode:
            logger.log("\n" + "="*50)
            logger.log("TEST MODE COMPLETED SUCCESSFULLY")
            logger.log("All functions working properly. Ready for full training!")
            logger.log("Run again without test mode for complete results.")
            logger.log("="*50)
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        logger.log(error_msg)
        print(error_msg)
        raise
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        logger.close()

if __name__ == '__main__':
    main()