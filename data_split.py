import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

unified_emotions = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fear': 5,
    'disgust': 6,
    'surprise': 7
}

ravdess_emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

savee_emotion_map = {
    'a': 'anger', 'd': 'disgust', 'f': 'fear', 'h': 'happiness',
    'n': 'neutral', 'sa': 'sadness', 'su': 'surprise'
}

crema_emotion_map = {
    'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

ravdess_to_unified = {
    'neutral': 'neutral', 'calm': 'calm', 'happy': 'happy', 'sad': 'sad',
    'angry': 'angry', 'fearful': 'fear', 'disgust': 'disgust', 'surprised': 'surprise'
}

savee_to_unified = {
    'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'happiness': 'happy',
    'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprise'
}

crema_to_unified = {
    'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'
}

tess_to_unified = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'pleasant_surprise': 'surprise',
    'pleasant_surprised': 'surprise'
}

def extract_ravdess_metadata(file_path):
    parts = os.path.basename(file_path).split('.')[0].split('-')
    emotion_code = parts[2]
    actor_id = parts[6]
    emotion = ravdess_emotion_map.get(emotion_code, 'unknown')
    unified_emotion = ravdess_to_unified.get(emotion, 'unknown')
    gender = 'male' if int(actor_id) % 2 == 1 else 'female'
    return {
        'file_path': file_path,
        'emotion': unified_emotion,
        'emotion_id': unified_emotions.get(unified_emotion, -1),
        'subdataset': 'Ravdess',
        'actor_id': f'Actor_{actor_id}',
        'gender': gender,
        'age_group': 'unknown',
        'split': None
    }

def extract_savee_metadata(file_path):
    filename = os.path.basename(file_path).split('.')[0]
    parts = filename.split('_')
    actor_id = parts[0]
    emotion_part = parts[1]
    match = re.match(r'([a-zA-Z]+)(\d+)', emotion_part)
    emotion_letter = match.group(1) if match else 'unknown'
    emotion = savee_emotion_map.get(emotion_letter, 'unknown')
    unified_emotion = savee_to_unified.get(emotion, 'unknown')
    return {
        'file_path': file_path,
        'emotion': unified_emotion,
        'emotion_id': unified_emotions.get(unified_emotion, -1),
        'subdataset': 'Savee',
        'actor_id': actor_id,
        'gender': 'male',
        'age_group': 'unknown',
        'split': None
    }

def extract_crema_metadata(file_path, actor_to_gender):
    parts = os.path.basename(file_path).split('.')[0].split('_')
    actor_id = parts[0]
    emotion_code = parts[2]
    emotion = crema_emotion_map.get(emotion_code, 'unknown')
    unified_emotion = crema_to_unified.get(emotion, 'unknown')
    gender = actor_to_gender.get(actor_id, 'unknown')
    return {
        'file_path': file_path,
        'emotion': unified_emotion,
        'emotion_id': unified_emotions.get(unified_emotion, -1),
        'subdataset': 'Crema',
        'actor_id': actor_id,
        'gender': gender,
        'age_group': 'unknown',
        'split': None
    }

def extract_tess_metadata(file_path):
    folder = os.path.basename(os.path.dirname(file_path))
    parts = folder.split('_')
    actress = parts[0]
    emotion = '_'.join(parts[1:]).lower()
    unified_emotion = tess_to_unified.get(emotion, 'unknown')
    age_group = 'older' if actress == 'OAF' else 'younger'
    return {
        'file_path': file_path,
        'emotion': unified_emotion,
        'emotion_id': unified_emotions.get(unified_emotion, -1),
        'subdataset': 'Tess',
        'actor_id': actress,
        'gender': 'female',
        'age_group': age_group,
        'split': None
    }

def collect_metadata(base_dir):
    metadata = []

    
    ravdess_dir = os.path.join(base_dir, 'ravdess_dataset')
    if os.path.exists(ravdess_dir):
        for actor_folder in os.listdir(ravdess_dir):
            actor_path = os.path.join(ravdess_dir, actor_folder)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(ravdess_dir, actor_folder, file)
                        metadata.append(extract_ravdess_metadata(file_path))
    
    
    savee_dir = os.path.join(base_dir, 'savee_dataset')
    if os.path.exists(savee_dir):
        for file in os.listdir(savee_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(savee_dir, file)
                metadata.append(extract_savee_metadata(file_path))

    
    crema_dir = os.path.join(base_dir, 'crema_dataset')
    actor_to_gender = {}  
    if os.path.exists(crema_dir):
        for file in os.listdir(crema_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(crema_dir, file)
                metadata.append(extract_crema_metadata(file_path, actor_to_gender))
    
    
    tess_dir = os.path.join(base_dir, 'tess_dataset')
    if os.path.exists(tess_dir):
        for folder in os.listdir(tess_dir):
            folder_path = os.path.join(tess_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(tess_dir, folder, file)
                        metadata.append(extract_tess_metadata(file_path))
    
    return pd.DataFrame(metadata)

def assign_splits(metadata):
    if metadata.empty:
        print("Metadata is empty, cannot assign splits.")
        return metadata
        
    
    ravdess_df = metadata[metadata['subdataset'] == 'Ravdess']
    if not ravdess_df.empty:
        ravdess_actors = ravdess_df['actor_id'].unique()
        male_actors = [a for a in ravdess_actors if ravdess_df[ravdess_df['actor_id'] == a]['gender'].iloc[0] == 'male']
        female_actors = [a for a in ravdess_actors if ravdess_df[ravdess_df['actor_id'] == a]['gender'].iloc[0] == 'female']
        
        
        if len(male_actors) >= 12 and len(female_actors) >= 12:
            train_male = np.random.choice(male_actors, 8, replace=False)
            train_female = np.random.choice(female_actors, 8, replace=False)
            remaining_male = [a for a in male_actors if a not in train_male]
            remaining_female = [a for a in female_actors if a not in train_female]
            val_male = np.random.choice(remaining_male, 2, replace=False)
            val_female = np.random.choice(remaining_female, 2, replace=False)
            test_male = [a for a in remaining_male if a not in val_male]
            test_female = [a for a in remaining_female if a not in val_female]
            
            train_actors = list(train_male) + list(train_female)
            val_actors = list(val_male) + list(val_female)
            test_actors = list(test_male) + list(test_female)
            
            metadata.loc[metadata['actor_id'].isin(train_actors), 'split'] = 'train'
            metadata.loc[metadata['actor_id'].isin(val_actors), 'split'] = 'val'
            metadata.loc[metadata['actor_id'].isin(test_actors), 'split'] = 'test'

    
    crema_df = metadata[metadata['subdataset'] == 'Crema']
    if not crema_df.empty:
        crema_actors = crema_df['actor_id'].unique()
        if len(crema_actors) > 0:
            train_actors, temp_actors = train_test_split(crema_actors, test_size=0.2, random_state=42)
            val_actors, test_actors = train_test_split(temp_actors, test_size=0.5, random_state=42)
            
            metadata.loc[metadata['actor_id'].isin(train_actors), 'split'] = 'train'
            metadata.loc[metadata['actor_id'].isin(val_actors), 'split'] = 'val'
            metadata.loc[metadata['actor_id'].isin(test_actors), 'split'] = 'test'

    
    for subdataset in ['Savee', 'Tess']:
        sub_df = metadata[metadata['subdataset'] == subdataset]
        if not sub_df.empty:
            actors = sub_df['actor_id'].unique()
            for actor in actors:
                actor_df = sub_df[sub_df['actor_id'] == actor]
                if len(actor_df) > 3: 
                    train_idx, temp_idx = train_test_split(actor_df.index, test_size=0.3, stratify=actor_df['emotion'], random_state=42)
                    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=actor_df.loc[temp_idx, 'emotion'], random_state=42)
                    metadata.loc[train_idx, 'split'] = 'train'
                    metadata.loc[val_idx, 'split'] = 'val'
                    metadata.loc[test_idx, 'split'] = 'test'
                else:
                    metadata.loc[actor_df.index, 'split'] = 'train'  

    return metadata

def main(base_dir):
    metadata = collect_metadata(base_dir)
    
    if metadata.empty:
        print("No metadata collected. Please check your directory structure and paths.")
        return

    metadata = assign_splits(metadata)

    
    if 'split' not in metadata.columns or metadata['split'].isnull().all():
        print("Warning: Splits could not be assigned. Check actor counts and data distribution.")
        metadata.to_csv('metadata_full.csv', index=False)
        print("Full metadata saved to metadata_full.csv")
    else:
        train_df = metadata[metadata['split'] == 'train']
        val_df = metadata[metadata['split'] == 'val']
        test_df = metadata[metadata['split'] == 'test']
        
        train_df.to_csv('train.csv', index=False)
        val_df.to_csv('val.csv', index=False)
        test_df.to_csv('test.csv', index=False)
        
        print("CSV files generated: train.csv, val.csv, test.csv")
        print("\n--- Split Counts ---")
        print(f"Train: {len(train_df)} files")
        print(f"Validation: {len(val_df)} files")
        print(f"Test: {len(test_df)} files")

if __name__ == '__main__':
    
    base_dir = '.' 
    main(base_dir)