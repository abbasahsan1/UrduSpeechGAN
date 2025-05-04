# -*- coding: utf-8 -*-
"""
TensorFlow/Keras script for Male-to-Female voice conversion using Mel spectrograms,
with data caching, checkpoint resuming, and integrated inference. Includes HiFi-GAN
vocoder support (training or loading) with Griffin-Lim fallback.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa
import glob
import argparse # Ensure this is imported
import pandas as pd
import soundfile as sf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, UpSampling1D, MaxPooling1D, LeakyReLU, Add, Lambda,
    Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D # <-- Added Cropping2D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys
from tensorflow.keras.utils import Sequence
import pickle  # For saving/loading processed data
import re      # For parsing checkpoint filenames
import traceback # For better error reporting

# --- Constants and Configuration ---
SAMPLE_RATE = 41000  # Sample rate from UrduSER dataset (rounded)
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE  # Max audio length in samples (10 seconds)
N_MELS = 80  # Number of Mel frequency bins
HOP_LENGTH = 256  # Hop length for STFT (~6ms at 41kHz, affects time resolution)
WIN_LENGTH = HOP_LENGTH * 4 # Window length for STFT (e.g., 1024)

# --- Paths ---
# !! IMPORTANT: Update these paths if necessary !!
DATASET_PATH = r"X:\VoiceChanger\UrduSER A comprehensive dataset for speech emotion recognition in Urdu language"
MODEL_SAVE_PATH = r"X:\VoiceChanger\models" # Where models and cache will be saved

# --- Derived Paths ---
METADATA_PATH = os.path.join(DATASET_PATH, "UrSEC_Description.csv")
DESC_PATH = os.path.join(DATASET_PATH, "desc.md")
PROCESSED_DATA_CACHE_PATH = os.path.join(MODEL_SAVE_PATH, "processed_data_cache.pkl")
VOCODER_CACHE_PATH = os.path.join(MODEL_SAVE_PATH, "vocoder_data_cache.pkl")
VOCODER_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "hifigan_vocoder_generator.h5") # Save generator specifically
CHECKPOINT_FILENAME_PATTERN = "voice_conversion_model_epoch_{epoch:02d}.h5" # For resumable checkpoints

# --- Behavior Flags ---
USE_PRETRAINED_VOCODER = True  # Try to load a vocoder model if it exists
USE_DATA_GENERATOR = True  # Highly recommended to avoid memory errors
# Calculate MAX_MEL_LENGTH dynamically and ensure it's an integer
MAX_MEL_LENGTH = int(np.ceil(MAX_AUDIO_LENGTH / HOP_LENGTH))
print(f"Derived MAX_MEL_LENGTH: {MAX_MEL_LENGTH}") # Print derived value

# --- Dataset Specific ---
EMOTIONS = ["Angry", "Fear", "Boredom", "Disgust", "Happy", "Neutral", "Sad"]
EMOTION_CODE_MAP = {
    "1": "Angry", "2": "Fear", "3": "Boredom", "4": "Disgust",
    "5": "Happy", "6": "Neutral", "7": "Sad"
}

# --- Ensure Save Directory Exists ---
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ==============================================================================
# == METADATA AND FILE FINDING FUNCTIONS (Based on Original Script) ==========
# ==============================================================================

def load_metadata():
    """Load and parse metadata from CSV and desc.md (best effort)."""
    dataset_info = {}
    metadata_df = None

    # --- Extract info from desc.md ---
    if os.path.exists(DESC_PATH):
        print(f"Reading dataset description from {DESC_PATH}")
        try:
            with open(DESC_PATH, 'r', encoding='utf-8') as f:
                desc_content = f.read()
            # Basic pattern and gender map from description
            dataset_info['filename_pattern'] = "[Actor_ID]_[Gender_Code]_[Emotion_Code]_[Sequence_Number].wav"
            dataset_info['gender_code_map'] = {"0": "male", "1": "female"}
            # Could add actor extraction here if needed
        except Exception as e:
            print(f"Warning: Error reading description file: {e}")
    else:
        print(f"Warning: Description file not found at {DESC_PATH}")
        # Provide defaults if desc.md is missing
        dataset_info['filename_pattern'] = "[Actor_ID]_[Gender_Code]_[Emotion_Code]_[Sequence_Number].wav"
        dataset_info['gender_code_map'] = {"0": "male", "1": "female"}


    # --- Load CSV metadata ---
    if os.path.exists(METADATA_PATH):
        print(f"Loading metadata from {METADATA_PATH}")
        try:
            metadata_df = pd.read_csv(METADATA_PATH, encoding='utf-8', dtype=str) # Read all as string initially
            print(f"Loaded metadata with {len(metadata_df)} raw entries.")

            # --- Attempt to map columns based on potential Urdu headers ---
            column_mapping = {}
            potential_headers = metadata_df.iloc[0].astype(str)
            header_row_found = False

            for col_original_name, header_val in potential_headers.items():
                if 'فائل' in header_val and ('نام' in header_val or 'Name' in header_val) : column_mapping[col_original_name] = 'file_name'; header_row_found=True
                elif 'جذبات' in header_val or 'Emotion' in header_val: column_mapping[col_original_name] = 'emotion_label'; header_row_found=True
                elif 'تشریح' in header_val or 'Transcript' in header_val : column_mapping[col_original_name] = 'transcription'; header_row_found=True
                elif 'صنف' in header_val or 'Gender' in header_val : column_mapping[col_original_name] = 'gender_label'; header_row_found=True
                elif ('اداکار' in header_val or 'Actor' in header_val) and ('نمبر' in header_val or 'ID' in header_val) : column_mapping[col_original_name] = 'actor_id'; header_row_found=True


            if header_row_found:
                print("Identified potential header row. Renaming columns:")
                metadata_df = metadata_df.iloc[1:].reset_index(drop=True) # Skip header row
                metadata_df = metadata_df.rename(columns=column_mapping)
                print("Mapped columns:", list(metadata_df.columns))
            else:
                 print("Warning: Could not confidently identify header row in CSV based on keywords.")


            # Store dataset info within the dataframe for easy access later
            metadata_df.dataset_info = dataset_info # This line might cause a UserWarning but is often okay
            print(f"Metadata processed. {len(metadata_df)} entries remain.")
            return metadata_df

        except Exception as e:
            print(f"Error loading or processing metadata CSV: {e}")
            traceback.print_exc()
            metadata_df = None

    else:
        print(f"Metadata file not found at {METADATA_PATH}")

    # --- Handle case where metadata_df couldn't be loaded ---
    if metadata_df is None:
         print("Metadata DataFrame is None. Will rely solely on file structure if possible.")
         # Attach dataset_info to a dummy object if needed by downstream functions,
         # but it's better to handle None explicitly.
         class InfoContainer: pass
         info_obj = InfoContainer()
         info_obj.dataset_info = dataset_info
         # Returning None is cleaner if handled properly later.
         # return info_obj # Alternative if needed
         return None

    # Fallback if loop completes unexpectedly (shouldn't happen)
    print("Proceeding without metadata CSV.")
    return None


def find_audio_file(file_name_base):
    """Find the full path of an audio file, searching common locations."""
    if not file_name_base: return None # Handle empty input

    if not file_name_base.lower().endswith('.wav'):
        file_name = file_name_base + '.wav'
    else:
        file_name = file_name_base

    search_paths = [
        os.path.join(DATASET_PATH, file_name),
    ]
    search_paths.extend([os.path.join(DATASET_PATH, emotion, file_name) for emotion in EMOTIONS])

    for path in search_paths:
        if os.path.exists(path):
            return path

    # print(f"Searching recursively for {file_name} in {DATASET_PATH}...") # Can be slow, enable if needed
    for root, _, files in os.walk(DATASET_PATH):
        if file_name in files:
            found_path = os.path.join(root, file_name)
            # print(f"Found at: {found_path}") # Verbose
            return found_path

    print(f"Warning: Audio file not found for base name: {file_name_base}")
    return None

def find_parallel_data_with_metadata(metadata):
    """Find parallel (male, female) audio file pairs using loaded metadata."""
    if metadata is None or 'file_name' not in metadata.columns:
        print("Metadata unavailable or missing 'file_name' column. Cannot use metadata approach.")
        return []

    parallel_pairs = []
    # Safely get dataset_info, provide default gender map if missing
    dataset_info = getattr(metadata, 'dataset_info', {})
    gender_map = dataset_info.get('gender_code_map', {"0": "male", "1": "female"})
    print("Attempting to find parallel pairs using metadata...")

    file_info = []
    processed_count = 0
    error_count = 0
    found_paths = 0

    print("Extracting info from filenames listed in metadata...")
    for idx, row in metadata.iterrows():
        file_name = row.get('file_name', '').strip()
        if not file_name:
            error_count += 1
            continue

        try:
            base_name = os.path.splitext(file_name)[0] # Handle extension safely
            parts = base_name.split('_')
            if len(parts) >= 4:
                actor_id, gender_code, emotion_code, sequence = parts[0], parts[1], parts[2], parts[3]

                if gender_code in gender_map and emotion_code in EMOTION_CODE_MAP:
                    gender = gender_map[gender_code]
                    emotion = EMOTION_CODE_MAP[emotion_code]
                    full_path = find_audio_file(file_name) # Use original name with extension potential
                    if full_path:
                        file_info.append({
                            'path': full_path, 'actor': actor_id, 'gender': gender,
                            'emotion': emotion, 'emotion_code': emotion_code, 'sequence': sequence,
                            'transcription': row.get('transcription')
                        })
                        found_paths += 1
                    else: error_count += 1
                else: error_count += 1
            else: error_count += 1
            processed_count += 1
        except Exception as e:
            print(f"Error processing metadata row {idx} for file {file_name}: {e}")
            error_count += 1

        if (processed_count) % 500 == 0 and processed_count > 0:
             print(f"  Processed {processed_count} metadata entries (Paths found: {found_paths}, Errors/Skipped: {error_count})")

    print(f"Finished filename analysis. Found {len(file_info)} valid entries with located audio files.")
    if not file_info: return []

    # Group by emotion and sequence
    grouped = {}
    for info in file_info:
        key = (info['emotion_code'], info['sequence'])
        if key not in grouped: grouped[key] = {'male': None, 'female': None, 'text': None}
        gender = info['gender']
        if gender == 'male' and grouped[key]['male'] is None:
            grouped[key]['male'] = info['path']
            grouped[key]['text'] = info.get('transcription')
        elif gender == 'female' and grouped[key]['female'] is None:
            grouped[key]['female'] = info['path']

    # Create final pairs list
    for key, paths in grouped.items():
        if paths['male'] and paths['female']:
            parallel_pairs.append((paths['male'], paths['female'], paths['text'], EMOTION_CODE_MAP.get(key[0])))

    print(f"Created {len(parallel_pairs)} parallel pairs based on matching emotion and sequence from metadata.")
    return parallel_pairs


def find_parallel_data_without_metadata():
    """Find parallel pairs based purely on filename convention if metadata fails."""
    print("Attempting to find parallel pairs using filename structure (metadata failed/unavailable)...")
    all_wav_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
    print(f"Found {len(all_wav_files)} total .wav files in dataset directory.")

    file_info = []
    gender_map = {"0": "male", "1": "female"}

    for file_path in all_wav_files:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        parts = base_name.split('_')
        if len(parts) >= 4:
            actor_id, gender_code, emotion_code, sequence = parts[0], parts[1], parts[2], parts[3]
            if gender_code in gender_map and emotion_code in EMOTION_CODE_MAP:
                file_info.append({
                    'path': file_path, 'gender': gender_map[gender_code],
                    'emotion_code': emotion_code, 'sequence': sequence
                })

    print(f"Found {len(file_info)} files matching the expected naming pattern.")
    if not file_info: return []

    # Group by emotion and sequence
    grouped = {}
    for info in file_info:
        key = (info['emotion_code'], info['sequence'])
        if key not in grouped: grouped[key] = {'male': None, 'female': None}
        gender = info['gender']
        if gender == 'male' and grouped[key]['male'] is None: grouped[key]['male'] = info['path']
        elif gender == 'female' and grouped[key]['female'] is None: grouped[key]['female'] = info['path']

    # Create final pairs list
    parallel_pairs = []
    for key, paths in grouped.items():
        if paths['male'] and paths['female']:
            parallel_pairs.append((paths['male'], paths['female'], None, EMOTION_CODE_MAP.get(key[0])))

    print(f"Created {len(parallel_pairs)} parallel pairs based on file structure matching.")
    return parallel_pairs


# ==============================================================================
# == AUDIO PROCESSING FUNCTIONS ================================================
# ==============================================================================

def extract_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """Extracts Log-Mel spectrogram from an audio waveform."""
    audio = audio.astype(np.float32)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=win_length,
        hop_length=hop_length, win_length=win_length, fmax=sr / 2
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def load_and_preprocess_audio(file_path, max_len_samples=MAX_AUDIO_LENGTH, trim_db=None):
    """Loads audio, pads/truncates, normalizes, and converts to Mel spectrogram."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        if trim_db is not None:
            audio, _ = librosa.effects.trim(audio, top_db=trim_db, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)

        if len(audio) > max_len_samples:
            audio = audio[:max_len_samples]
        else:
            audio = np.pad(audio, (0, max_len_samples - len(audio)), mode='constant')

        peak = np.max(np.abs(audio))
        if peak > 1e-5: audio = audio / peak * 0.95

        mel_spec = extract_mel_spectrogram(audio, sr)

        # Ensure correct time dimension length (MAX_MEL_LENGTH)
        current_mel_len = mel_spec.shape[1]
        if current_mel_len > MAX_MEL_LENGTH:
            mel_spec = mel_spec[:, :MAX_MEL_LENGTH]
        elif current_mel_len < MAX_MEL_LENGTH:
            pad_width = MAX_MEL_LENGTH - current_mel_len
            min_db = np.min(mel_spec)
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_db)

        return mel_spec.astype(np.float32)

    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return None


# ==============================================================================
# == DATA PREPARATION WITH CACHING =============================================
# ==============================================================================

def prepare_data(force_reprocess=False):
    """Prepare data for voice conversion model training, using cache."""
    global PROCESSED_DATA_CACHE_PATH

    if not force_reprocess and os.path.exists(PROCESSED_DATA_CACHE_PATH):
        print(f"Attempting to load processed data from cache: {PROCESSED_DATA_CACHE_PATH}")
        try:
            with open(PROCESSED_DATA_CACHE_PATH, 'rb') as f: cached_data = pickle.load(f)
            if 'train' in cached_data and 'val' in cached_data:
                print(f"Successfully loaded data from cache: {len(cached_data['train'])} train, {len(cached_data['val'])} val pairs.")
                return cached_data['train'], cached_data['val']
            else: print("Cache file structure invalid. Reprocessing...")
        except Exception as e: print(f"Error loading cached data: {e}. Reprocessing...")

    print("Processing data from scratch...")
    metadata = load_metadata()
    file_pairs = find_parallel_data_with_metadata(metadata) if metadata is not None else []
    if not file_pairs:
        print("Could not find pairs using metadata. Trying file structure...")
        file_pairs = find_parallel_data_without_metadata()
    if not file_pairs: raise ValueError("Could not find any parallel audio file pairs.")

    print(f"Found {len(file_pairs)} raw male-female file pairs.")
    print("Converting audio pairs to Mel spectrograms...")
    mf_pairs = []
    processed_count, error_count = 0, 0

    for i, (male_file, female_file, text, emotion) in enumerate(file_pairs):
        male_mel_spec = load_and_preprocess_audio(male_file)
        female_mel_spec = load_and_preprocess_audio(female_file)
        if male_mel_spec is not None and female_mel_spec is not None and \
           male_mel_spec.shape == (N_MELS, MAX_MEL_LENGTH) and \
           female_mel_spec.shape == (N_MELS, MAX_MEL_LENGTH):
            mf_pairs.append((male_mel_spec, female_mel_spec))
            processed_count += 1
        else:
            error_count += 1
            # Optionally print why it failed (None or shape mismatch)
            # print(f"Skipping pair {i}...")

        if (processed_count + error_count) % 50 == 0 and (processed_count + error_count) > 0:
             print(f"  Processed {processed_count + error_count}/{len(file_pairs)} pairs (Ok: {processed_count}, Err: {error_count})")

    print(f"Finished processing. Successfully created {len(mf_pairs)} valid spectrogram pairs.")
    if not mf_pairs: raise ValueError("No valid spectrogram pairs created.")

    train_pairs, val_pairs = train_test_split(mf_pairs, test_size=0.1, random_state=42)
    print(f"Split data: {len(train_pairs)} training, {len(val_pairs)} validation.")

    print(f"Saving processed data to cache: {PROCESSED_DATA_CACHE_PATH}")
    try:
        with open(PROCESSED_DATA_CACHE_PATH, 'wb') as f:
            pickle.dump({'train': train_pairs, 'val': val_pairs}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Successfully saved data to cache.")
    except Exception as e: print(f"Error saving data cache: {e}")

    return train_pairs, val_pairs


# ==============================================================================
# == DATA GENERATOR ============================================================
# ==============================================================================

class DataGenerator(Sequence):
    """Generates batches of (male_mel, female_mel) pairs for Keras model.fit."""
    def __init__(self, pairs, batch_size=8, shuffle=True):
        self.pairs = [p for p in pairs if p[0] is not None and p[1] is not None] # Filter out None pairs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_pairs = [self.pairs[k] for k in batch_indexes]

        X_batch = np.zeros((self.batch_size, N_MELS, MAX_MEL_LENGTH, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, N_MELS, MAX_MEL_LENGTH, 1), dtype=np.float32)

        for i, (male_mel, female_mel) in enumerate(batch_pairs):
             # Basic check, prepare_data should ensure this already
             if male_mel.shape == (N_MELS, MAX_MEL_LENGTH) and female_mel.shape == (N_MELS, MAX_MEL_LENGTH):
                 X_batch[i, :, :, 0] = male_mel
                 y_batch[i, :, :, 0] = female_mel
             else: # Log if unexpected shape slips through
                  print(f"DataGen Warning: Unexpected shape at batch {index}, item {i}. M:{male_mel.shape}, F:{female_mel.shape}")

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)


# ==============================================================================
# == VOICE CONVERSION MODEL ARCHITECTURE =======================================
# ==============================================================================

def build_model(input_shape=(N_MELS, MAX_MEL_LENGTH, 1)):
    """Builds the U-Net like Conv2D voice conversion model."""
    print(f"Building voice conversion model with input shape: {input_shape}")
    height, width, channels = input_shape # Unpack for clarity

    encoder_input = Input(shape=input_shape, name="encoder_input")

    # --- Encoder ---
    e1 = Conv2D(64, (3, 3), padding='same', name='enc_conv1_1')(encoder_input)
    e1 = LeakyReLU(negative_slope=0.2)(e1) # Use negative_slope
    e1 = Conv2D(64, (3, 3), padding='same', name='enc_conv1_2')(e1)
    e1 = LeakyReLU(negative_slope=0.2)(e1)
    p1 = MaxPooling2D((2, 2), padding='same', name='pool1')(e1)

    e2 = Conv2D(128, (3, 3), padding='same', name='enc_conv2_1')(p1)
    e2 = LeakyReLU(negative_slope=0.2)(e2)
    e2 = Conv2D(128, (3, 3), padding='same', name='enc_conv2_2')(e2)
    e2 = LeakyReLU(negative_slope=0.2)(e2)
    p2 = MaxPooling2D((2, 2), padding='same', name='pool2')(e2)

    e3 = Conv2D(256, (3, 3), padding='same', name='enc_conv3_1')(p2)
    e3 = LeakyReLU(negative_slope=0.2)(e3)
    e3 = Conv2D(256, (3, 3), padding='same', name='enc_conv3_2')(e3)
    e3 = LeakyReLU(negative_slope=0.2)(e3)
    p3 = MaxPooling2D((2, 2), padding='same', name='pool3')(e3)

    # --- Bottleneck ---
    b = Conv2D(512, (3, 3), padding='same', name='bottleneck_conv1')(p3)
    b = LeakyReLU(negative_slope=0.2)(b)
    b = Conv2D(512, (3, 3), padding='same', name='bottleneck_conv2')(b)
    b = LeakyReLU(negative_slope=0.2)(b)

    # --- Decoder ---
    u3 = UpSampling2D((2, 2), name='upsample3')(b)
    # Optional: Add skip connection: u3 = Concatenate()([u3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', name='dec_conv3_1')(u3)
    d3 = LeakyReLU(negative_slope=0.2)(d3)
    d3 = Conv2D(256, (3, 3), padding='same', name='dec_conv3_2')(d3)
    d3 = LeakyReLU(negative_slope=0.2)(d3)

    u2 = UpSampling2D((2, 2), name='upsample2')(d3)
    # Optional: Add skip connection: u2 = Concatenate()([u2, e2])
    d2 = Conv2D(128, (3, 3), padding='same', name='dec_conv2_1')(u2)
    d2 = LeakyReLU(negative_slope=0.2)(d2)
    d2 = Conv2D(128, (3, 3), padding='same', name='dec_conv2_2')(d2)
    d2 = LeakyReLU(negative_slope=0.2)(d2)

    u1 = UpSampling2D((2, 2), name='upsample1')(d2) # Shape: (None, height, width_up1, 128)
    # Optional: Add skip connection: u1 = Concatenate()([u1, e1])
    d1 = Conv2D(64, (3, 3), padding='same', name='dec_conv1_1')(u1)
    d1 = LeakyReLU(negative_slope=0.2)(d1)
    d1 = Conv2D(64, (3, 3), padding='same', name='dec_conv1_2')(d1)
    d1 = LeakyReLU(negative_slope=0.2)(d1) # Shape: (None, height, width_up1, 64)

    # --- *** FIX: Add Cropping Layer *** ---
    # Calculate cropping needed for the width (time) dimension
    current_width = d1.shape[2] # Width after last decoder block (e.g., 1608)
    target_width = input_shape[1] # Original input width (e.g., 1602)

    if current_width is None or target_width is None:
         print("Warning: Cannot determine cropping automatically due to dynamic shapes.")
         # Fallback or raise error - for now assume cropping might not be needed or use a fixed guess
         cropped_d1 = d1
    elif current_width > target_width:
        crop_total = current_width - target_width
        crop_left = crop_total // 2
        crop_right = crop_total - crop_left
        print(f"Cropping decoder output width from {current_width} to {target_width} (left: {crop_left}, right: {crop_right})")
        cropped_d1 = Cropping2D(cropping=((0, 0), (crop_left, crop_right)), name='output_cropping')(d1)
    elif current_width < target_width:
         print(f"Warning: Decoder output width {current_width} is smaller than target {target_width}. Padding might be needed or check architecture.")
         # Potentially add padding here if necessary, but cropping is more common for U-Nets
         cropped_d1 = d1 # Pass through without cropping if smaller (might error later)
    else:
        # Widths match, no cropping needed
        cropped_d1 = d1

    # --- Output layer ---
    # Applied to the (potentially) cropped tensor
    output = Conv2D(1, (1, 1), activation='linear', padding='same', name='output_mel')(cropped_d1)

    # Create and compile model
    model = Model(inputs=encoder_input, outputs=output, name="VoiceConversionModel")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')
    print("Voice conversion model summary:")
    model.summary(line_length=120) # Wider summary lines

    return model

# ==============================================================================
# == CHECKPOINT HANDLING =======================================================
# ==============================================================================

def find_latest_checkpoint(model_dir):
    """Finds the latest model checkpoint file based on epoch number."""
    checkpoint_pattern = os.path.join(model_dir, CHECKPOINT_FILENAME_PATTERN.split('{')[0] + '*.h5') # More robust glob
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files: return None, -1

    latest_epoch = -1
    latest_checkpoint_path = None
    for f in checkpoint_files:
        match = re.search(r"epoch_(\d+)\.h5$", os.path.basename(f))
        if match:
            try:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint_path = f
            except ValueError: continue

    if latest_checkpoint_path: print(f"Found latest checkpoint: {os.path.basename(latest_checkpoint_path)} (Epoch {latest_epoch})")
    else: print("No valid checkpoint files matching the pattern found.")
    return latest_checkpoint_path, latest_epoch


# ==============================================================================
# == VOICE CONVERSION MODEL TRAINING ===========================================
# ==============================================================================

def train_model(epochs=50, batch_size=8):
    """Trains the voice conversion model, handling data, resuming, callbacks."""
    train_pairs, val_pairs = prepare_data(force_reprocess=args.force_reprocess_data) # Use args flag directly
    if not train_pairs or not val_pairs:
        print("Error: No training or validation data available.")
        return None, None
    print(f"Data ready: {len(train_pairs)} train, {len(val_pairs)} val pairs.")

    latest_checkpoint, last_epoch = find_latest_checkpoint(MODEL_SAVE_PATH)
    initial_epoch = 0
    model = None

    if latest_checkpoint:
        print(f"Attempting to load model from: {latest_checkpoint}")
        try:
            model = load_model(latest_checkpoint)
            initial_epoch = last_epoch + 1
            print(f"Model loaded successfully. Resuming training from epoch {initial_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint '{os.path.basename(latest_checkpoint)}': {e}")
            print("Building a new model from scratch...")
            traceback.print_exc()
            model = None

    if model is None:
        # Calculate input shape based on actual data from prepare_data
        # Ensures model matches the preprocessed data dimensions
        sample_mel_shape = train_pairs[0][0].shape # (N_MELS, MAX_MEL_LENGTH)
        model_input_shape = (sample_mel_shape[0], sample_mel_shape[1], 1)
        model = build_model(input_shape=model_input_shape)
        initial_epoch = 0

    if not hasattr(model, 'optimizer') or not model.optimizer:
         print("Model loaded without optimizer or wasn't compiled. Compiling...")
         model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')

    checkpoint_filepath = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILENAME_PATTERN)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False,
        save_best_only=False, monitor='val_loss', verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    callbacks_list = [model_checkpoint_callback, early_stopping_callback]

    if initial_epoch >= epochs:
        print(f"Training already completed ({initial_epoch}/{epochs} epochs). Loading latest model.")
        return model, None

    print(f"\n--- Starting/Resuming Training ---")
    print(f"Target Epochs: {epochs}, Initial Epoch: {initial_epoch}, Batch Size: {batch_size}")
    print(f"Using Data Generator: {USE_DATA_GENERATOR}")

    history = None
    if USE_DATA_GENERATOR:
        train_generator = DataGenerator(train_pairs, batch_size=batch_size, shuffle=True)
        val_generator = DataGenerator(val_pairs, batch_size=batch_size, shuffle=False)
        print(f"Training steps: {len(train_generator)}, Validation steps: {len(val_generator)}")
        history = model.fit(
            train_generator, epochs=epochs, initial_epoch=initial_epoch,
            validation_data=val_generator, callbacks=callbacks_list
        )
    else: # Non-generator path
        print("WARNING: Training without data generator.")
        try:
            print("Loading all data into memory...")
            X_train = np.array([p[0] for p in train_pairs], dtype=np.float32)[:, :, :, np.newaxis]
            y_train = np.array([p[1] for p in train_pairs], dtype=np.float32)[:, :, :, np.newaxis]
            X_val = np.array([p[0] for p in val_pairs], dtype=np.float32)[:, :, :, np.newaxis]
            y_val = np.array([p[1] for p in val_pairs], dtype=np.float32)[:, :, :, np.newaxis]
            print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
            history = model.fit(
                X_train, y_train, epochs=epochs, initial_epoch=initial_epoch,
                batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list
            )
        except MemoryError:
            print("\n*** MEMORY ERROR: Failed to load all data into RAM. Use the data generator. ***")
            return None, None
        except Exception as e:
            print(f"Unexpected error during non-generator training: {e}")
            traceback.print_exc()
            return None, None

    if history:
        print("Training finished.")
        if early_stopping_callback.stopped_epoch > 0: print(f"Early stopping triggered at epoch {early_stopping_callback.stopped_epoch + 1}")
        print("Model training process completed.")
    elif initial_epoch < epochs:
         print("Training did not run or failed.")
         return None, None

    return model, history


# ==============================================================================
# == HIFIGAN VOCODER IMPLEMENTATION (Simplified) ===============================
# ==============================================================================

def ResBlock(x, channels, kernel_size=3, dilation_rates=[1, 3, 5]):
    """Simplified residual block for HiFi-GAN."""
    shortcut = x
    for dilation in dilation_rates:
        x_in = x # Store input for this conv pair
        x = LeakyReLU(negative_slope=0.1)(x) # Use negative_slope
        x = Conv1D(channels, kernel_size, dilation_rate=dilation, padding='same')(x)
        x = LeakyReLU(negative_slope=0.1)(x) # Use negative_slope
        x = Conv1D(channels, kernel_size, dilation_rate=1, padding='same')(x) # Second conv often dilation 1
        x = Add()([x, x_in]) # Add skip within the res conv pairs
    # The original paper might have a slightly different structure (e.g., applying conv before leakyrelu)
    # Check paper/implementations for exact details if needed
    return x # Return processed block (shortcut is handled outside if needed)

def build_hifigan_generator(input_length=MAX_MEL_LENGTH, n_mels=N_MELS, upsample_initial_channel=512,
                           upsample_rates=[8, 8, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4]):
    """Builds a simplified HiFi-GAN Generator model using Conv1DTranspose."""
    print("Building HiFi-GAN Generator...")
    mel_input = Input(shape=(input_length, n_mels), name="hifigan_mel_input")

    x = Conv1D(upsample_initial_channel, kernel_size=7, padding='same', name='gen_initial_conv')(mel_input)

    current_channels = upsample_initial_channel
    for i, (u_rate, u_kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        current_channels //= 2
        x = LeakyReLU(negative_slope=0.1)(x) # Use negative_slope
        x = tf.keras.layers.Conv1DTranspose(
            filters=current_channels, kernel_size=u_kernel, strides=u_rate,
            padding='same', name=f'gen_upsample_{i}'
        )(x)

        # Apply Residual Blocks
        res_stack_out = x
        for j in range(3): # 3 ResBlocks per stack
             res_stack_out = ResBlock(res_stack_out, channels=current_channels)
        x = res_stack_out # Update main flow with output of ResBlocks


    x = LeakyReLU(negative_slope=0.1)(x) # Use negative_slope
    output_waveform = Conv1D(1, kernel_size=7, padding='same', activation='tanh', name='gen_output_conv')(x)
    output_waveform = tf.keras.layers.Reshape((-1,), name='gen_output_reshape')(output_waveform)

    generator = Model(inputs=mel_input, outputs=output_waveform, name="HiFiGAN_Generator")
    print("HiFi-GAN Generator Summary:")
    generator.summary(line_length=100)
    return generator

def build_hifigan_discriminator(name="HiFiGAN_Discriminator"):
    """Builds a simplified Multi-Scale Discriminator inspired by HiFi-GAN."""
    print("Building Simplified Multi-Scale Discriminator...")
    audio_input = Input(shape=(None, 1), name="disc_audio_input")

    outputs = [] # Store final output of each scale + feature maps
    feature_outputs = []
    x = audio_input
    filters = [16, 64, 128, 256, 512, 1024, 1024] # Example filter progression
    kernel_sizes = [15, 41, 41, 41, 41, 5, 3] # Example kernel sizes
    strides = [1, 2, 2, 4, 4, 1, 1] # Strides control downsampling

    for i in range(len(filters)):
        x = Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], strides=strides[i], padding='same')(x)
        x = LeakyReLU(negative_slope=0.2)(x) # Use negative_slope
        feature_outputs.append(x) # Store feature map

    # Final convolution for classification (output is a map, not single value yet)
    x = Conv1D(1, kernel_size=3, padding='same')(x)
    feature_outputs.append(x) # Add final output map

    # The model can output all feature maps or just the final map
    # For basic loss, final map is often enough
    discriminator = Model(inputs=audio_input, outputs=feature_outputs, name=name) # Output list of feature maps
    print("Simplified Discriminator Summary:")
    discriminator.summary(line_length=100)
    return discriminator


# --- GAN Loss Functions ---
def hifigan_discriminator_loss(disc_real_outputs, disc_fake_outputs):
    """Calculates LSGAN loss for discriminator outputs (list of tensors)."""
    loss = 0.0
    # Use only the *final* output map from each discriminator branch/scale
    real_final_out = disc_real_outputs[-1]
    fake_final_out = disc_fake_outputs[-1]
    loss += tf.reduce_mean(tf.square(real_final_out - 1.0)) # Real = 1
    loss += tf.reduce_mean(tf.square(fake_final_out - 0.0)) # Fake = 0
    return loss

def hifigan_generator_adversarial_loss(disc_fake_outputs):
    """Calculates LSGAN adversarial loss for generator (wants fake to be 1)."""
    loss = 0.0
    # Use only the *final* output map
    fake_final_out = disc_fake_outputs[-1]
    loss += tf.reduce_mean(tf.square(fake_final_out - 1.0))
    return loss

def hifigan_feature_matching_loss(disc_real_outputs, disc_fake_outputs):
    """Calculates L1 Feature Matching Loss between intermediate D layers."""
    loss = 0.0
    # Iterate through all layers *except* the final output layer
    num_layers = len(disc_real_outputs) - 1
    if num_layers <= 0: return 0.0 # No intermediate layers to match

    for i in range(num_layers):
        loss += tf.reduce_mean(tf.abs(disc_real_outputs[i] - disc_fake_outputs[i]))
    return loss / num_layers # Average L1 loss across layers


# --- Vocoder Data Preparation and Training ---
def prepare_vocoder_data(force_reprocess=False):
    """Prepares (mel, audio) pairs for vocoder training, uses cache."""
    global VOCODER_CACHE_PATH
    if not force_reprocess and os.path.exists(VOCODER_CACHE_PATH):
        print(f"Loading vocoder data from cache: {VOCODER_CACHE_PATH}")
        try:
            with open(VOCODER_CACHE_PATH, 'rb') as f: cached_data = pickle.load(f)
            if 'train' in cached_data and 'val' in cached_data:
                print(f"Loaded vocoder cache: {len(cached_data['train'])} train, {len(cached_data['val'])} val.")
                return cached_data['train'], cached_data['val']
            else: print("Invalid vocoder cache structure.")
        except Exception as e: print(f"Error loading vocoder cache: {e}.")
        print("Reprocessing vocoder data...")

    print("Preparing data for vocoder training from scratch...")
    all_wav_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
    print(f"Found {len(all_wav_files)} audio files for vocoder.")
    vocoder_pairs = []
    processed_count, error_count = 0, 0
    files_to_process = all_wav_files

    for i, audio_file in enumerate(files_to_process):
        try:
            audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
            if len(audio) > MAX_AUDIO_LENGTH: audio = audio[:MAX_AUDIO_LENGTH]
            else: audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)), mode='constant')
            peak = np.max(np.abs(audio));
            if peak > 1e-5: audio = audio / peak * 0.95
            mel_spec = extract_mel_spectrogram(audio, sr)
            if mel_spec.shape[1] != MAX_MEL_LENGTH: # Ensure correct length after processing
                if mel_spec.shape[1] > MAX_MEL_LENGTH: mel_spec = mel_spec[:, :MAX_MEL_LENGTH]
                else:
                    pad_width = MAX_MEL_LENGTH - mel_spec.shape[1]
                    min_db = np.min(mel_spec); mel_spec = np.pad(mel_spec, ((0,0), (0, pad_width)), mode='constant', constant_values=min_db)
            vocoder_pairs.append((mel_spec.T.astype(np.float32), audio.astype(np.float32))) # Mel (Time, Mels)
            processed_count += 1
        except Exception as e: error_count += 1; print(f"Error processing {os.path.basename(audio_file)} for vocoder: {e}")
        if (processed_count + error_count) % 100 == 0 and (processed_count + error_count) > 0:
            print(f"  Processed {processed_count + error_count}/{len(files_to_process)} files (Ok: {processed_count}, Err: {error_count})")

    print(f"Created {len(vocoder_pairs)} (mel, audio) pairs.")
    if not vocoder_pairs: raise ValueError("No vocoder data pairs created.")
    train_pairs, val_pairs = train_test_split(vocoder_pairs, test_size=0.05, random_state=42)

    print(f"Saving vocoder data to cache: {VOCODER_CACHE_PATH}")
    try:
        with open(VOCODER_CACHE_PATH, 'wb') as f: pickle.dump({'train': train_pairs, 'val': val_pairs}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Vocoder data saved to cache.")
    except Exception as e: print(f"Error saving vocoder data cache: {e}")
    return train_pairs, val_pairs

class VocoderDataGenerator(Sequence):
    """Generates batches of (mel, audio) for vocoder training."""
    def __init__(self, pairs, batch_size):
        self.pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self): return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_pairs = [self.pairs[k] for k in batch_indexes]
        X_mel = np.zeros((self.batch_size, MAX_MEL_LENGTH, N_MELS), dtype=np.float32)
        y_audio = np.zeros((self.batch_size, MAX_AUDIO_LENGTH), dtype=np.float32)
        for i, (mel, audio) in enumerate(batch_pairs):
            mel_len = min(mel.shape[0], MAX_MEL_LENGTH); audio_len = min(audio.shape[0], MAX_AUDIO_LENGTH)
            X_mel[i, :mel_len, :] = mel[:mel_len, :]
            y_audio[i, :audio_len] = audio[:audio_len]
        return X_mel, y_audio[:, :, np.newaxis] # Add channel dim for audio

    def on_epoch_end(self): np.random.shuffle(self.indexes)


# --- Simplified Vocoder Training Loop ---
def train_vocoder(epochs=100, batch_size=16):
    """Trains the HiFi-GAN vocoder (simplified GAN loop)."""
    print("\n--- Starting Vocoder Training ---")
    train_pairs, val_pairs = prepare_vocoder_data(force_reprocess=args.force_reprocess_data)
    train_gen = VocoderDataGenerator(train_pairs, batch_size)

    generator = build_hifigan_generator(input_length=MAX_MEL_LENGTH)
    discriminator = build_hifigan_discriminator()

    gen_optimizer = Adam(learning_rate=1e-4, beta_1=0.8, beta_2=0.99) # HiFiGAN often uses specific betas
    disc_optimizer = Adam(learning_rate=1e-4, beta_1=0.8, beta_2=0.99)
    lambda_fm = 2.0 # Weight for feature matching loss

    steps_per_epoch = len(train_gen)
    for epoch in range(epochs):
        print(f"\nVocoder Epoch {epoch+1}/{epochs}")
        progbar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=['G_loss', 'D_loss', 'FM_loss', 'Adv_loss'])
        epoch_g_loss, epoch_d_loss, epoch_fm_loss, epoch_adv_loss = 0.0, 0.0, 0.0, 0.0

        for step, (mel_batch, audio_batch) in enumerate(train_gen):
            # --- Train Discriminator ---
            with tf.GradientTape() as disc_tape:
                fake_audio = generator(mel_batch, training=True)
                # Get list of layer outputs
                real_outputs_list = discriminator(audio_batch, training=True)
                fake_outputs_list = discriminator(fake_audio, training=True)
                d_loss = hifigan_discriminator_loss(real_outputs_list, fake_outputs_list)
            disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

            # --- Train Generator ---
            with tf.GradientTape() as gen_tape:
                fake_audio_gen = generator(mel_batch, training=True)
                # Get discriminator outputs again for G training
                real_outputs_gen = discriminator(audio_batch, training=False) # No D training here
                fake_outputs_gen = discriminator(fake_audio_gen, training=True)
                # Adversarial Loss
                adv_loss = hifigan_generator_adversarial_loss(fake_outputs_gen)
                # Feature Matching Loss
                fm_loss = hifigan_feature_matching_loss(real_outputs_gen, fake_outputs_gen)
                # Total Generator Loss
                g_loss = adv_loss + lambda_fm * fm_loss

            gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

            # Update metrics
            epoch_g_loss += g_loss.numpy(); epoch_d_loss += d_loss.numpy();
            epoch_fm_loss += fm_loss.numpy(); epoch_adv_loss += adv_loss.numpy();
            progbar.update(step + 1, values=[
                ("G_loss", g_loss), ("D_loss", d_loss),
                ("FM_loss", fm_loss), ("Adv_loss", adv_loss)])

            if step >= steps_per_epoch - 1: break

        avg_g = epoch_g_loss / steps_per_epoch; avg_d = epoch_d_loss / steps_per_epoch;
        avg_fm = epoch_fm_loss / steps_per_epoch; avg_adv = epoch_adv_loss / steps_per_epoch;
        print(f" Epoch {epoch+1} Summary - G: {avg_g:.4f}, D: {avg_d:.4f}, FM: {avg_fm:.4f}, Adv: {avg_adv:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            generator.save(VOCODER_MODEL_PATH, save_format="h5")
            print(f"Saved Vocoder Generator to {VOCODER_MODEL_PATH}")

    print("Vocoder training finished.")
    return generator


_vocoder_instance = None
def get_vocoder(force_train=False):
    """Loads the trained vocoder generator or initiates training."""
    global _vocoder_instance, VOCODER_MODEL_PATH, USE_PRETRAINED_VOCODER
    if _vocoder_instance is not None and not force_train: return _vocoder_instance

    if not force_train and USE_PRETRAINED_VOCODER and os.path.exists(VOCODER_MODEL_PATH):
        print(f"Loading existing vocoder generator from: {VOCODER_MODEL_PATH}")
        try:
            # Need to register custom layers/objects if HiFi-GAN uses them
            # For this simplified version, direct loading might work
            _vocoder_instance = load_model(VOCODER_MODEL_PATH, compile=False)
            print("Vocoder loaded successfully.")
            return _vocoder_instance
        except Exception as e: print(f"Error loading vocoder: {e}. Will attempt training.")
        _vocoder_instance = None

    if force_train or _vocoder_instance is None:
        print("Attempting to train a new vocoder...")
        _vocoder_instance = train_vocoder(epochs=args.vocoder_epochs, batch_size=args.vocoder_batch_size) # Use args

    return _vocoder_instance


# ==============================================================================
# == MEL TO AUDIO SYNTHESIS (Vocoder or Griffin-Lim) ===========================
# ==============================================================================

def mel_to_audio(mel_spectrogram, use_griffin_lim=False):
    """Converts Mel spectrogram (N_MELS, Time) back to audio waveform."""
    print("Synthesizing audio from Mel spectrogram...")
    vocoder = None
    if not use_griffin_lim: vocoder = get_vocoder(force_train=args.train_vocoder_only) # Trigger training only if that mode is set

    if vocoder is not None:
        try:
            print("Using HiFi-GAN vocoder...")
            # Ensure input is (Batch, Time, Mels)
            mel_input = mel_spectrogram.T[np.newaxis, :, :].astype(np.float32)
            generated_audio = vocoder.predict(mel_input)[0]
            print("Vocoder synthesis successful.")
            return generated_audio.astype(np.float32)
        except Exception as e: print(f"Vocoder inference error: {e}. Falling back to Griffin-Lim."); traceback.print_exc()

    print("Using Griffin-Lim algorithm...")
    n_iter_gl = 60
    try:
        # Ensure input is (N_MELS, Time)
        if mel_spectrogram.shape[1] == N_MELS: mel_spectrogram = mel_spectrogram.T
        mel_power = librosa.db_to_power(mel_spectrogram, ref=np.max)
        stft_mag = librosa.feature.inverse.mel_to_stft(mel_power, sr=SAMPLE_RATE, n_fft=WIN_LENGTH)
        audio_gl = librosa.griffinlim(stft_mag, n_iter=n_iter_gl, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        print(f"Griffin-Lim synthesis done ({n_iter_gl} iterations).")
        return audio_gl.astype(np.float32)
    except Exception as e:
         print(f"Griffin-Lim synthesis error: {e}"); traceback.print_exc()
         return np.zeros(MAX_AUDIO_LENGTH, dtype=np.float32)


# ==============================================================================
# == VOICE CONVERSION FUNCTION =================================================
# ==============================================================================

def convert_voice(conversion_model, input_file, output_file, force_griffin_lim=False):
    """Loads input, converts using model, synthesizes output, saves."""
    print(f"\n--- Starting Voice Conversion ---"); print(f"Input:  {input_file}"); print(f"Output: {output_file}")
    try:
        print("Loading/preprocessing input...")
        input_mel = load_and_preprocess_audio(input_file)
        if input_mel is None: print("Failed to process input. Aborting."); return
        print(f"Input Mel Shape: {input_mel.shape}")

        model_input = input_mel[np.newaxis, :, :, np.newaxis].astype(np.float32)
        print(f"Running conversion model (input shape: {model_input.shape})...")
        output_mel_tensor = conversion_model.predict(model_input)
        output_mel = np.squeeze(output_mel_tensor)
        print(f"Output Mel Shape: {output_mel.shape}")

        converted_audio = mel_to_audio(output_mel, use_griffin_lim=force_griffin_lim)

        peak = np.max(np.abs(converted_audio));
        if peak > 1e-5: converted_audio = converted_audio / peak * 0.95

        print(f"Saving converted audio to {output_file} (SR: {SAMPLE_RATE})...")
        sf.write(output_file, converted_audio, SAMPLE_RATE)
        print("Conversion successful!")
    except Exception as e: print(f"\n*** ERROR DURING CONVERSION for {input_file}: ***\n{traceback.format_exc()}\n{'*' * 60}")


# ==============================================================================
# == MAIN EXECUTION AND COMMAND-LINE INTERFACE =================================
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Male-to-Female Voice Conversion model or Convert Audio.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--convert-only', action='store_true', help='Run in conversion mode. Requires --input-file.')
    parser.add_argument('--train-only', action='store_true', help='Run only voice conversion model training.')
    parser.add_argument('--train-vocoder-only', action='store_true', help='Run only vocoder training.')
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=50, help='Epochs for voice conversion model.')
    train_group.add_argument('--batch-size', type=int, default=8, help='Batch size for voice conversion model.')
    train_group.add_argument('--force-reprocess-data', action='store_true', help='Ignore data cache and re-process audio.')
    train_group.add_argument('--no-generator', action='store_true', help='Disable data generator (needs huge RAM).')
    train_group.add_argument('--vocoder-epochs', type=int, default=100, help='Epochs for vocoder training.')
    train_group.add_argument('--vocoder-batch-size', type=int, default=16, help='Batch size for vocoder training.')
    convert_group = parser.add_argument_group('Conversion Parameters')
    convert_group.add_argument('--input-file', type=str, default=None, help='Input audio (.wav) for conversion.')
    convert_group.add_argument('--output-file', type=str, default=None, help='Output audio (.wav) for conversion.')
    convert_group.add_argument('--model-checkpoint', type=str, default=None, help='Specific .h5 checkpoint for conversion.')
    convert_group.add_argument('--use-griffin-lim', action='store_true', help='Force Griffin-Lim for synthesis.')
    args = parser.parse_args()

    if args.no_generator: USE_DATA_GENERATOR = False; print("WARNING: Data generator disabled.")

    run_conv_train = not (args.convert_only or args.train_vocoder_only)
    run_voc_train = not (args.convert_only or args.train_only)
    run_inference = args.convert_only

    if args.train_vocoder_only: run_voc_train = True # Ensure vocoder training runs if specified

    # --- Execute ---
    conv_model = None
    if run_conv_train:
        print("\n" + "="*20 + " Training Voice Conversion Model " + "="*20)
        conv_model, history = train_model(epochs=args.epochs, batch_size=args.batch_size)
        if conv_model is None: print("Voice conversion training failed."); sys.exit(1)
        print("="*70)

    if run_voc_train:
        print("\n" + "="*20 + " Training Vocoder Model " + "="*20)
        get_vocoder(force_train=True) # Triggers training
        print("="*60)

    if run_inference:
        print("\n" + "="*20 + " Running Voice Conversion " + "="*20)
        if not args.input_file or not os.path.exists(args.input_file): parser.error("--input-file required and must exist for conversion.")
        model_to_use = None
        if args.model_checkpoint:
            if os.path.exists(args.model_checkpoint): model_path = args.model_checkpoint
            else: parser.error(f"Specified checkpoint not found: {args.model_checkpoint}")
        else:
            model_path, _ = find_latest_checkpoint(MODEL_SAVE_PATH)
            if not model_path: parser.error(f"No model found in {MODEL_SAVE_PATH}. Train or use --model-checkpoint.")
        print(f"Loading conversion model: {model_path}")
        try:
             # Load potentially without compiling if issues persist, then re-compile
             model_to_use = load_model(model_path, compile=False)
             model_to_use.compile(optimizer=Adam(learning_rate=1e-4), loss='mae') # Re-compile is safe
             print("Model loaded.")
        except Exception as e: print(f"Error loading model {model_path}: {e}"); sys.exit(1)

        output_path = args.output_file
        if not output_path:
             base, ext = os.path.splitext(os.path.basename(args.input_file)); output_path = os.path.join(MODEL_SAVE_PATH, f"{base}_converted{ext}")
             print(f"Output path defaulted to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        convert_voice(model_to_use, args.input_file, output_path, args.use_griffin_lim)
        print("="*60)

    print("\nScript finished.")