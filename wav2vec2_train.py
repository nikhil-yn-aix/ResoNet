import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification, 
    AutoConfig,
    logging as transformers_logging
)
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import warnings
from datetime import datetime
import gc


warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()


SAMPLE_RATE = 16000
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
        self.log_file = open(os.path.join(log_dir, 'wav2vec2_experiment.log'), 'w', buffering=1)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + '\n')
        self.log_file.flush()
        
    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

def load_audio_for_wav2vec2(file_path, target_duration=2.5):
    
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        if len(waveform) == 0:
            raise ValueError(f"Empty audio file: {file_path}")
        
        
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        
        target_length = int(target_duration * SAMPLE_RATE)
        if len(waveform) < target_length:
            
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            
            start = (len(waveform) - target_length) // 2
            waveform = waveform[start:start + target_length]
        
        return waveform.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        
        return np.zeros(int(target_duration * SAMPLE_RATE), dtype=np.float32)

class Wav2Vec2EmotionDataset(Dataset):
    def __init__(self, csv_path, target_duration=2.5):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        self.target_duration = target_duration
        
        
        valid_samples = []
        for _, row in self.data.iterrows():
            if os.path.exists(row['file_path']):
                valid_samples.append(row)
        
        self.samples = valid_samples
        if not self.samples:
            raise ValueError(f"No valid audio files found in {csv_path}")
        
        print(f"Loaded {len(self.samples)} valid samples from {csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        
        waveform = load_audio_for_wav2vec2(sample['file_path'], self.target_duration)
        
        return {
            'waveform': torch.tensor(waveform, dtype=torch.float32),
            'emotion_id': sample['emotion_id'],
            'subdataset': sample['subdataset']
        }

class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_feature_encoder=True):
        super().__init__()
        
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = NUM_CLASSES
        
        
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        
        if freeze_feature_encoder:
            self.wav2vec2.wav2vec2.feature_extractor._freeze_parameters()
            print("Feature encoder frozen for efficient training")
        
        
        hidden_size = config.hidden_size  
        print(f"Wav2Vec2 config hidden size: {hidden_size}")
        
        
        
        original_classifier = self.wav2vec2.classifier
        if hasattr(original_classifier, 'in_features'):
            actual_input_size = original_classifier.in_features
        else:
            
            if isinstance(original_classifier, nn.Sequential):
                actual_input_size = original_classifier[0].in_features
            else:
                actual_input_size = hidden_size
        
        print(f"Actual classifier input size: {actual_input_size}")
        
        
        self.wav2vec2.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(actual_input_size, 512),  
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, NUM_CLASSES)
        )
        
        print(f"Custom classifier created with input size: {actual_input_size}")
        
    def forward(self, input_values):
        return self.wav2vec2(input_values).logits

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2.0):
        super().__init__()
        
        
        weights = torch.zeros(NUM_CLASSES)
        for emotion, weight in class_weights.items():
            idx = EMOTION_CLASSES.index(emotion)
            weights[idx] = weight
        
        self.class_weights = weights
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.class_weights.to(inputs.device), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, logger):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        waveforms = batch['waveform'].to(device, non_blocking=True)
        labels = batch['emotion_id'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        
        with autocast('cuda'):
            try:
                logits = model(waveforms)
                loss = criterion(logits, labels)
            except RuntimeError as e:
                logger.log(f"ERROR: Runtime error in forward pass: {str(e)}")
                logger.log(f"Waveform shape: {waveforms.shape}")
                logger.log(f"Labels shape: {labels.shape}")
                raise
        
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.log(f"WARNING: Invalid loss {loss.item():.4f} at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
        
        
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, macro_f1, weighted_f1

def validate_epoch(model, dataloader, criterion, device, logger):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_subdatasets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            waveforms = batch['waveform'].to(device, non_blocking=True)
            labels = batch['emotion_id'].to(device, non_blocking=True)
            subdatasets = batch['subdataset']
            
            with autocast('cuda'):
                try:
                    logits = model(waveforms)
                    loss = criterion(logits, labels)
                except RuntimeError as e:
                    print(f"ERROR in validation: {str(e)}")
                    print(f"Waveform shape: {waveforms.shape}")
                    raise
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subdatasets.extend(subdatasets)
    
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    
    per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    
    per_dataset_acc = {}
    for dataset in set(all_subdatasets):
        dataset_mask = [i for i, d in enumerate(all_subdatasets) if d == dataset]
        if dataset_mask:
            dataset_preds = [all_predictions[i] for i in dataset_mask]
            dataset_labels = [all_labels[i] for i in dataset_mask]
            per_dataset_acc[dataset] = accuracy_score(dataset_labels, dataset_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels,
        'per_class_f1': per_class_f1,
        'per_dataset_acc': per_dataset_acc
    }

def train_wav2vec2_model(train_loader, val_loader, test_loader, device, epochs, results_dir, logger):
    
    model = Wav2Vec2EmotionClassifier().to(device)
    
    
    criterion = WeightedFocalLoss(CLASS_WEIGHTS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.999))
    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    
    scaler = GradScaler('cuda')
    
    
    best_f1 = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_weighted_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_weighted_f1': [], 'lr': []
    }
    
    logger.log(f"Starting Wav2Vec2 training for {epochs} epochs")
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        logger.log(f"\nEpoch {epoch+1}/{epochs}")
        
        
        train_loss, train_acc, train_f1, train_wf1 = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, logger
        )
        
        
        val_metrics = validate_epoch(model, val_loader, criterion, device, logger)
        
        
        scheduler.step(val_metrics['macro_f1'])
        
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_weighted_f1'].append(train_wf1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, os.path.join(results_dir, 'best_wav2vec2_model.pth'))
            logger.log(f"New best model saved with F1: {best_f1:.4f}")
        
        
        logger.log(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.log(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")
        
        
        torch.cuda.empty_cache()
        gc.collect()
    
    
    checkpoint = torch.load(os.path.join(results_dir, 'best_wav2vec2_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.log("\nEvaluating on test set...")
    test_metrics = validate_epoch(model, test_loader, criterion, device, logger)
    
    logger.log(f"\nFinal Test Results:")
    logger.log(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.log(f"F1-Macro: {test_metrics['macro_f1']:.4f}")
    logger.log(f"F1-Weighted: {test_metrics['weighted_f1']:.4f}")
    
    return test_metrics, history, model

def create_visualizations(test_metrics, history, results_dir, logger):
    logger.log("Creating visualizations...")
    
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    
    axes[0, 1].plot(epochs_range, history['train_acc'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    
    axes[1, 0].plot(epochs_range, history['train_f1'], 'b-', label='Training', linewidth=2)
    axes[1, 0].plot(epochs_range, history['val_f1'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('Training and Validation F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    
    axes[1, 1].plot(epochs_range, history['lr'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wav2vec2_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
                cbar_kws={'shrink': 0.8})
    plt.title(f'Wav2Vec2 Confusion Matrix\nAccuracy: {test_metrics["accuracy"]:.3f}, F1-Macro: {test_metrics["macro_f1"]:.3f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wav2vec2_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(EMOTION_CLASSES, test_metrics['per_class_f1'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(EMOTION_CLASSES))), alpha=0.8)
    plt.title('Wav2Vec2 Per-Class F1 Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    
    for bar, score in zip(bars, test_metrics['per_class_f1']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wav2vec2_per_class_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.log("Visualizations saved successfully")

def save_results(test_metrics, history, results_dir, logger):
    logger.log("Saving detailed results...")
    
    
    results = {
        'experiment_info': {
            'model': 'Wav2Vec2-base',
            'timestamp': datetime.now().isoformat(),
            'total_epochs': len(history['train_loss']),
            'best_epoch': np.argmax(history['val_f1']) + 1,
            'emotion_classes': EMOTION_CLASSES,
            'class_weights': CLASS_WEIGHTS
        },
        'test_results': {
            'accuracy': float(test_metrics['accuracy']),
            'macro_f1': float(test_metrics['macro_f1']),
            'weighted_f1': float(test_metrics['weighted_f1']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'per_class_f1': test_metrics['per_class_f1'].tolist(),
            'per_dataset_accuracy': test_metrics['per_dataset_acc']
        },
        'training_history': history,
        'predictions': test_metrics['predictions'],
        'labels': test_metrics['labels']
    }
    
    
    with open(os.path.join(results_dir, 'wav2vec2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    
    summary_df = pd.DataFrame([{
        'Model': 'Wav2Vec2',
        'Accuracy': test_metrics['accuracy'],
        'F1-Macro': test_metrics['macro_f1'],
        'F1-Weighted': test_metrics['weighted_f1']
    }])
    summary_df.to_csv(os.path.join(results_dir, 'wav2vec2_summary.csv'), index=False)
    
    
    per_class_df = pd.DataFrame({
        'Emotion': EMOTION_CLASSES,
        'F1-Score': test_metrics['per_class_f1']
    })
    per_class_df.to_csv(os.path.join(results_dir, 'wav2vec2_per_class.csv'), index=False)
    
    logger.log("Results saved successfully")
    
    return results

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'wav2vec2_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    logger = Logger(results_dir)
    
    try:
        
        required_files = ['train.csv', 'val.csv', 'test.csv']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
        
        logger.log("Starting Wav2Vec2 baseline experiment...")
        logger.log(f"Results will be saved to: {results_dir}")
        
        
        logger.log("Loading datasets...")
        train_dataset = Wav2Vec2EmotionDataset('train.csv')
        val_dataset = Wav2Vec2EmotionDataset('val.csv')
        test_dataset = Wav2Vec2EmotionDataset('test.csv')
        
        
        batch_size = 8  
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        logger.log(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        logger.log(f"Batch size: {batch_size}")
        
        
        epochs = 25  
        test_metrics, history, model = train_wav2vec2_model(
            train_loader, val_loader, test_loader, device, epochs, results_dir, logger
        )
        
        
        create_visualizations(test_metrics, history, results_dir, logger)
        
        
        results = save_results(test_metrics, history, results_dir, logger)
        
        
        logger.log("\n" + "="*60)
        logger.log("WAV2VEC2 BASELINE EXPERIMENT COMPLETED")
        logger.log("="*60)
        logger.log(f"Final Test Results:")
        logger.log(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        logger.log(f"  F1-Macro:    {test_metrics['macro_f1']:.4f}")
        logger.log(f"  F1-Weighted: {test_metrics['weighted_f1']:.4f}")
        logger.log(f"\nResults saved in: {results_dir}")
        logger.log("Key files:")
        logger.log("  - wav2vec2_results.json: Complete results")
        logger.log("  - wav2vec2_training_history.png: Training curves")
        logger.log("  - wav2vec2_confusion_matrix.png: Confusion matrix")
        logger.log("  - wav2vec2_per_class_f1.png: Per-class performance")
        logger.log("  - best_wav2vec2_model.pth: Best model checkpoint")
        
        return results
        
    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        raise
    finally:
        logger.close()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()