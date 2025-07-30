import os
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
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.append('.')

from resonet import EmotionRecognitionModel, EmotionDataset
from baseline_ablation_train import SimpleCNN, CNNLSTM, NoAttentionResoNet, NoGRUResoNet, NoResNetResoNet, CachedDataset

try:
    from wav2vec2_train import Wav2Vec2EmotionClassifier, Wav2Vec2EmotionDataset as Wav2Vec2Dataset
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False

EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def mcnemar_test(y_true, pred1, pred2):
    correct1 = (pred1 == y_true).astype(int)
    correct2 = (pred2 == y_true).astype(int)
    b = np.sum((correct1 == 1) & (correct2 == 0))
    c = np.sum((correct1 == 0) & (correct2 == 1))
    if (b + c) == 0:
        return 1.0, 0.0
    chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    return p_value, chi2_stat

def bootstrap_confidence_interval(predictions, labels, metric='f1_macro', n_bootstrap=1000):
    n_samples = len(predictions)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        if metric == 'accuracy':
            score = accuracy_score(boot_labels, boot_preds)
        elif metric == 'f1_macro':
            score = f1_score(boot_labels, boot_preds, average='macro', zero_division=0)
        elif metric == 'f1_weighted':
            score = f1_score(boot_labels, boot_preds, average='weighted', zero_division=0)
        bootstrap_scores.append(score)
    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    return {
        'mean': float(np.mean(bootstrap_scores)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std': float(np.std(bootstrap_scores))
    }

def analyze_per_dataset_performance(predictions, labels, datasets):
    unique_datasets = np.unique(datasets)
    per_dataset_results = {}
    for dataset in unique_datasets:
        dataset_mask = datasets == dataset
        dataset_preds = predictions[dataset_mask]
        dataset_labels = labels[dataset_mask]
        if len(dataset_preds) > 0:
            accuracy = accuracy_score(dataset_labels, dataset_preds)
            macro_f1 = f1_score(dataset_labels, dataset_preds, average='macro', zero_division=0)
            weighted_f1 = f1_score(dataset_labels, dataset_preds, average='weighted', zero_division=0)
            per_dataset_results[dataset] = {
                'n_samples': len(dataset_preds),
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1
            }
    return per_dataset_results

def initialize_dynamic_layers(model, sample_input, device):
    
    model.eval()
    with torch.no_grad():
        try:
            if hasattr(model, 'forward'):
                
                _ = model(sample_input)
            else:
                
                _ = model(sample_input)
        except Exception as e:
            log_message(f"Warning: Could not initialize dynamic layers: {e}")

def load_model_and_predict(model_path, model_class, test_loader, device, model_type='standard'):
    log_message(f"Loading {model_type} model: {model_path}")
    if not os.path.exists(model_path):
        log_message(f"ERROR: Model file not found: {model_path}")
        return None, None, None
    
    try:
        model = model_class().to(device)
        
        
        sample_batch = next(iter(test_loader))
        if model_type == 'wav2vec2':
            sample_input = sample_batch['waveform'][:1].to(device)
        elif model_type == 'cached':
            sample_input = sample_batch[0][:1].to(device)
        else:
            sample_input = sample_batch[0][:1].to(device)
        
        
        initialize_dynamic_layers(model, sample_input, device)
        
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            log_message(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            log_message(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        model.eval()
        log_message(f"Successfully loaded {model_path} (with {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys)")
        
        all_predictions = []
        all_labels = []
        all_datasets = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Testing {os.path.basename(model_path)}", leave=False):
                try:
                    if model_type == 'wav2vec2':
                        inputs = batch_data['waveform'].to(device, non_blocking=True)
                        labels = batch_data['emotion_id']
                        datasets = batch_data['subdataset']
                        outputs = model(inputs)
                    elif model_type == 'cached':
                        inputs, labels, datasets = batch_data
                        inputs = inputs.to(device, non_blocking=True)
                        
                        model_output = model(inputs)
                        if isinstance(model_output, tuple):
                            outputs, _ = model_output
                        else:
                            outputs = model_output
                    else:
                        
                        inputs, labels = batch_data
                        
                        if not hasattr(load_model_and_predict, '_dataset_info'):
                            
                            import pandas as pd
                            test_df = pd.read_csv('test.csv')
                            load_model_and_predict._dataset_info = test_df['subdataset'].values
                        
                        
                        batch_size = len(labels)
                        start_idx = getattr(load_model_and_predict, '_batch_idx', 0)
                        end_idx = start_idx + batch_size
                        datasets = load_model_and_predict._dataset_info[start_idx:end_idx].tolist()
                        load_model_and_predict._batch_idx = end_idx
                        
                        inputs = inputs.to(device, non_blocking=True)
                        model_output = model(inputs)
                        if isinstance(model_output, tuple):
                            outputs, _ = model_output
                        else:
                            outputs = model_output
                    
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.numpy() if torch.is_tensor(labels) else labels)
                    all_datasets.extend(datasets)
                    
                except Exception as e:
                    log_message(f"Error processing batch: {e}")
                    continue
        
        if not all_predictions:
            log_message(f"ERROR: No predictions generated for {model_path}")
            return None, None, None
        
        
        if hasattr(load_model_and_predict, '_batch_idx'):
            delattr(load_model_and_predict, '_batch_idx')
            
        return np.array(all_predictions), np.array(all_labels), np.array(all_datasets)
        
    except Exception as e:
        log_message(f"ERROR loading {model_path}: {str(e)}")
        return None, None, None

def create_per_dataset_table(model_results, results_dir):
    log_message("Creating per-dataset analysis...")
    all_datasets = set()
    for results in model_results.values():
        if 'per_dataset_results' in results:
            all_datasets.update(results['per_dataset_results'].keys())
    all_datasets = sorted(list(all_datasets))
    
    if not all_datasets:
        log_message("No per-dataset results to analyze")
        return
    
    for metric in ['accuracy', 'macro_f1']:
        table_data = []
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            if metric == 'accuracy':
                row['Overall'] = f"{results['accuracy']:.4f}"
            else:
                row['Overall'] = f"{results['macro_f1']:.4f}"
            
            if 'per_dataset_results' in results:
                for dataset in all_datasets:
                    if dataset in results['per_dataset_results']:
                        dataset_result = results['per_dataset_results'][dataset]
                        row[dataset] = f"{dataset_result[metric]:.4f}"
                    else:
                        row[dataset] = "N/A"
            else:
                for dataset in all_datasets:
                    row[dataset] = "N/A"
            table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            metric_name = 'Accuracy' if metric == 'accuracy' else 'F1_Macro'
            df.to_csv(os.path.join(results_dir, f'per_dataset_{metric_name.lower()}.csv'), index=False)
            log_message(f"Per-dataset {metric_name} table:")
            print(df.to_string(index=False))
            print()

def create_comparison_plot(model_results, results_dir):
    model_names = list(model_results.keys())
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    f1_scores = [model_results[name]['macro_f1'] for name in model_names]
    
    
    if all('confidence_intervals' in model_results[name] for name in model_names):
        acc_cis = [model_results[name]['confidence_intervals']['accuracy'] for name in model_names]
        f1_cis = [model_results[name]['confidence_intervals']['f1_macro'] for name in model_names]
        
        acc_errors = [[acc - ci['ci_lower'] for acc, ci in zip(accuracies, acc_cis)],
                      [ci['ci_upper'] - acc for acc, ci in zip(accuracies, acc_cis)]]
        f1_errors = [[f1 - ci['ci_lower'] for f1, ci in zip(f1_scores, f1_cis)],
                     [ci['ci_upper'] - f1 for f1, ci in zip(f1_scores, f1_cis)]]
    else:
        acc_errors = None
        f1_errors = None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.8)
    if acc_errors:
        ax1.errorbar(range(len(model_names)), accuracies, yerr=acc_errors, fmt='none', color='black', capsize=5)
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    bars2 = ax2.bar(model_names, f1_scores, color='lightcoral', alpha=0.8)
    if f1_errors:
        ax2.errorbar(range(len(model_names)), f1_scores, yerr=f1_errors, fmt='none', color='black', capsize=5)
    ax2.set_title('Model F1-Macro Comparison', fontweight='bold')
    ax2.set_ylabel('F1-Macro Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")

    results_dir = f'model_comparison'
    os.makedirs(results_dir, exist_ok=True)
    
    
    use_cached_datasets = True
    try:
        
        if os.path.exists('feature_cache_info.json'):
            log_message("Using cached datasets for comparison")
        else:
            log_message("Cached features not found, using regular datasets")
            use_cached_datasets = False
    except:
        use_cached_datasets = False
    
    model_configs = {
        'ResoNet': {'class': EmotionRecognitionModel, 'path': 'best_ResoNet.pth', 'type': 'standard'},
        'NoAttentionResoNet': {'class': NoAttentionResoNet, 'path': 'best_NoAttentionResoNet.pth', 'type': 'cached'},
        'NoGRUResoNet': {'class': NoGRUResoNet, 'path': 'best_NoGRUResoNet.pth', 'type': 'cached'},
        'NoResNetResoNet': {'class': NoResNetResoNet, 'path': 'best_NoResNetResoNet.pth', 'type': 'cached'},
        'SimpleCNN': {'class': SimpleCNN, 'path': 'best_SimpleCNN.pth', 'type': 'cached'},
        'CNNLSTM': {'class': CNNLSTM, 'path': 'best_CNNLSTM.pth', 'type': 'cached'},
    }
    
    if WAV2VEC2_AVAILABLE:
        model_configs['Wav2Vec2'] = {'class': Wav2Vec2EmotionClassifier, 'path': 'best_wav2vec2_model.pth', 'type': 'wav2vec2'}
    
    model_results = {}
    
    for model_name, config in model_configs.items():
        log_message(f"\nTesting {model_name}...")
        
        
        if config['type'] == 'wav2vec2' and WAV2VEC2_AVAILABLE:
            test_dataset = Wav2Vec2Dataset('test.csv')
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        elif config['type'] == 'cached' and use_cached_datasets:
            try:
                test_dataset = CachedDataset('test.csv', feature_type='deep')
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
            except Exception as e:
                log_message(f"Failed to load cached dataset for {model_name}: {e}")
                log_message("Falling back to regular dataset")
                test_dataset = EmotionDataset('test.csv', augment=False, delta=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
                config['type'] = 'standard'
        else:
            test_dataset = EmotionDataset('test.csv', augment=False, delta=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        predictions, labels, datasets = load_model_and_predict(
            config['path'], config['class'], test_loader, device, config['type']
        )
        
        if predictions is not None and labels is not None:
            accuracy = accuracy_score(labels, predictions)
            macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(labels, predictions)
            per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
            
            
            ci_accuracy = bootstrap_confidence_interval(predictions, labels, 'accuracy')
            ci_f1_macro = bootstrap_confidence_interval(predictions, labels, 'f1_macro')
            ci_f1_weighted = bootstrap_confidence_interval(predictions, labels, 'f1_weighted')
            
            
            per_dataset_results = {}
            if datasets is not None and len(set(datasets)) > 1:
                per_dataset_results = analyze_per_dataset_performance(predictions, labels, datasets)
            
            model_results[model_name] = {
                'predictions': predictions,
                'labels': labels,
                'datasets': datasets,
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'confusion_matrix': conf_matrix,
                'per_class_f1': per_class_f1,
                'per_dataset_results': per_dataset_results,
                'confidence_intervals': {
                    'accuracy': ci_accuracy,
                    'f1_macro': ci_f1_macro,
                    'f1_weighted': ci_f1_weighted
                }
            }
            
            log_message(f"{model_name} Results:")
            log_message(f"  Overall - Accuracy: {accuracy:.4f}, F1-Macro: {macro_f1:.4f}")
            if per_dataset_results:
                log_message(f"  Per-dataset results:")
                for dataset, results in per_dataset_results.items():
                    log_message(f"    {dataset}: Acc={results['accuracy']:.4f}, F1={results['macro_f1']:.4f} (n={results['n_samples']})")
        else:
            log_message(f"Failed to get predictions for {model_name}")
    
    if len(model_results) < 2:
        log_message("ERROR: Need at least 2 successful models for comparison")
        return
    
    log_message(f"\nSuccessfully tested {len(model_results)} models")
    log_message("\nRunning pairwise statistical tests...")
    
    model_names = list(model_results.keys())
    pairwise_results = {}
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                pred1 = model_results[model1]['predictions']
                pred2 = model_results[model2]['predictions']
                labels = model_results[model1]['labels']
                
                
                if len(pred1) != len(pred2) or len(pred1) != len(labels):
                    log_message(f"Warning: Mismatched prediction lengths for {model1} vs {model2}")
                    continue
                
                mcnemar_p, mcnemar_chi2 = mcnemar_test(labels, pred1, pred2)
                f1_diff = model_results[model1]['macro_f1'] - model_results[model2]['macro_f1']
                
                pair_key = f"{model1}_vs_{model2}"
                pairwise_results[pair_key] = {
                    'model1': model1,
                    'model2': model2,
                    'mcnemar_p_value': mcnemar_p,
                    'mcnemar_chi2': mcnemar_chi2,
                    'f1_difference': f1_diff,
                    'significant': mcnemar_p < 0.05
                }
                
                significance = "SIGNIFICANT" if mcnemar_p < 0.05 else "Not significant"
                log_message(f"  {model1} vs {model2}: p={mcnemar_p:.4f} ({significance})")
    
    
    summary_data = []
    for model_name, results in model_results.items():
        ci_acc = results['confidence_intervals']['accuracy']
        ci_f1 = results['confidence_intervals']['f1_macro']
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Accuracy_CI': f"[{ci_acc['ci_lower']:.3f}, {ci_acc['ci_upper']:.3f}]",
            'F1_Macro': f"{results['macro_f1']:.4f}",
            'F1_Macro_CI': f"[{ci_f1['ci_lower']:.3f}, {ci_f1['ci_upper']:.3f}]",
            'F1_Weighted': f"{results['weighted_f1']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['F1_Macro_numeric'] = [results['macro_f1'] for results in model_results.values()]
    summary_df = summary_df.sort_values('F1_Macro_numeric', ascending=False).drop('F1_Macro_numeric', axis=1)
    summary_df.to_csv(os.path.join(results_dir, 'model_comparison_summary.csv'), index=False)
    
    
    if pairwise_results:
        sig_data = []
        for pair_key, pair_data in pairwise_results.items():
            sig_data.append({
                'Comparison': pair_key,
                'Model1': pair_data['model1'],
                'Model2': pair_data['model2'],
                'McNemar_p_value': f"{pair_data['mcnemar_p_value']:.4f}",
                'F1_Difference': f"{pair_data['f1_difference']:.4f}",
                'Significant': 'Yes' if pair_data['significant'] else 'No'
            })
        sig_df = pd.DataFrame(sig_data)
        sig_df.to_csv(os.path.join(results_dir, 'statistical_tests.csv'), index=False)
    
    
    create_per_dataset_table(model_results, results_dir)
    create_comparison_plot(model_results, results_dir)
    
    
    results_for_json = {}
    for model_name, results in model_results.items():
        results_for_json[model_name] = {
            'accuracy': float(results['accuracy']),
            'macro_f1': float(results['macro_f1']),
            'weighted_f1': float(results['weighted_f1']),
            'predictions': results['predictions'].tolist(),
            'labels': results['labels'].tolist(),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'per_class_f1': results['per_class_f1'].tolist(),
            'per_dataset_results': results['per_dataset_results'],
            'confidence_intervals': results['confidence_intervals']
        }
    
    with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
        json.dump({
            'model_results': results_for_json,
            'pairwise_comparisons': pairwise_results,
            'summary': summary_df.to_dict('records')
        }, f, indent=2, cls=NumpyEncoder)
    
    log_message(f"\nResults saved to: {results_dir}")
    log_message("\nSUMMARY:")
    print(summary_df.to_string(index=False))
    
    if pairwise_results:
        significant_pairs = [pair for pair, data in pairwise_results.items() if data['significant']]
        log_message(f"\nStatistically significant differences: {len(significant_pairs)}")
        for pair in significant_pairs:
            log_message(f"  {pair}: p={pairwise_results[pair]['mcnemar_p_value']:.4f}")

if __name__ == '__main__':
    compare_models()