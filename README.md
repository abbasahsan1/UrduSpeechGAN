# Urdu Male-to-Female Voice Conversion

This project implements a deep learning model using TensorFlow/Keras to convert male speech in Urdu to female speech. It utilizes Mel spectrograms as the intermediate representation and includes an integrated HiFi-GAN vocoder (with Griffin-Lim as a fallback) for synthesizing audio from the converted spectrograms.

## Features

*   **Male-to-Female Voice Conversion:** Core functionality using a U-Net like Convolutional Neural Network (Conv2D).
*   **Mel Spectrograms:** Uses Mel spectrograms for audio representation during conversion.
*   **HiFi-GAN Vocoder:** Includes training and inference support for a HiFi-GAN vocoder to generate high-fidelity audio from Mel spectrograms.
*   **Griffin-Lim Fallback:** Provides the Griffin-Lim algorithm as an alternative for audio synthesis if the vocoder is unavailable or fails.
*   **Data Caching:** Preprocesses and caches audio data (Mel spectrograms) to speed up subsequent training runs.
*   **Checkpoint Resuming:** Automatically finds and resumes training from the latest saved model checkpoint.
*   **Integrated Inference:** Allows direct conversion of audio files using a trained model.
*   **Command-Line Interface:** Provides easy control over training modes, inference, and parameters via command-line arguments.
*   **Dataset Flexibility:** Attempts to load metadata from CSV and parse filenames, with fallback to filename structure if metadata is missing.

## Dataset

This script is designed to work with the **UrduSER** dataset ("A comprehensive dataset for speech emotion recognition in Urdu language").

*   **You need to download this dataset separately.**
*   The script expects the dataset to be located at the path specified by `DATASET_PATH`.
*   It specifically looks for a metadata file `UrSEC_Description.csv` within the dataset directory, although it has fallback mechanisms if the CSV is missing or structured differently.
*   The script relies on a specific filename pattern (`[Actor_ID]_[Gender_Code]_[Emotion_Code]_[Sequence_Number].wav`) to find parallel male/female audio files, especially if the metadata parsing fails.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   `pip` (Python package installer)
    *   Git (optional, for cloning)

2.  **Clone the Repository (Optional):**
    If you have the script as part of a Git repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
    Otherwise, ensure you have the Python script (`your_script_name.py`) and the `requirements.txt` file in your working directory.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Use the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have `requirements.txt`, create it with the content from the previous response).*

5.  **Configure Paths:**
    *   **CRITICAL:** Open the Python script (`your_script_name.py`) in a text editor.
    *   Locate the `--- Paths ---` section.
    *   **Update `DATASET_PATH`:** Change the path to point to the **root directory** where you extracted the UrduSER dataset.
    *   **Update `MODEL_SAVE_PATH`:** Change this path to the desired location where trained models, checkpoints, and data caches will be saved.
    *   Example:
        ```python
        # --- Paths ---
        # !! IMPORTANT: Update these paths if necessary !!
        DATASET_PATH = r"C:\path\to\your\UrduSER_Dataset" # Use your actual path
        MODEL_SAVE_PATH = r"C:\path\to\save\models_and_cache" # Use your actual path
        ```

6.  **GPU Support (Optional but Recommended):**
    *   For significantly faster training, ensure you have a compatible NVIDIA GPU, the necessary NVIDIA drivers, CUDA Toolkit, and cuDNN library installed.
    *   Follow the official TensorFlow GPU setup guide for your operating system and hardware: [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)

## Usage

The script is controlled via command-line arguments.

### 1. Training the Voice Conversion Model

This trains the primary U-Net like model that converts male Mel spectrograms to female Mel spectrograms.

```bash
python your_script_name.py --train-only --epochs 50 --batch-size 8
```

*   `--train-only`: Ensures only the voice conversion model is trained.
*   `--epochs <number>`: Sets the number of training epochs (default: 50).
*   `--batch-size <number>`: Sets the batch size (default: 8). Adjust based on your GPU memory.
*   `--force-reprocess-data`: Add this flag if you want to ignore the cache and re-process all audio files from the dataset. Useful if you changed preprocessing steps or the dataset.
*   `--no-generator`: (Use with caution!) Disables the data generator and loads all data into RAM. Only feasible with very small datasets and large amounts of RAM.

Training progress, loss, and validation loss will be printed. Checkpoints (`voice_conversion_model_epoch_XX.h5`) will be saved in the `MODEL_SAVE_PATH`. Training will automatically resume from the latest checkpoint if found.

### 2. Training the HiFi-GAN Vocoder

This trains the vocoder model, which learns to synthesize audio waveforms from Mel spectrograms. This is independent of the voice conversion model training but uses the same audio data. **Note:** Vocoder training can be computationally expensive and time-consuming.

```bash
python your_script_name.py --train-vocoder-only --vocoder-epochs 100 --vocoder-batch-size 16
```

*   `--train-vocoder-only`: Ensures only the HiFi-GAN vocoder is trained.
*   `--vocoder-epochs <number>`: Sets the number of epochs for vocoder training (default: 100).
*   `--vocoder-batch-size <number>`: Sets the batch size for vocoder training (default: 16).
*   `--force-reprocess-data`: This will also force reprocessing of the vocoder data cache (`vocoder_data_cache.pkl`).

The trained vocoder generator will be saved as `hifigan_vocoder_generator.h5` in the `MODEL_SAVE_PATH`.

### 3. Performing Voice Conversion (Inference)

This uses a trained voice conversion model and a vocoder (or Griffin-Lim) to convert an input male audio file to a female version.

```bash
python your_script_name.py --convert-only --input-file /path/to/input_male.wav --output-file /path/to/output_female.wav
```

*   `--convert-only`: Puts the script into inference mode.
*   `--input-file <path>`: **Required.** Path to the input male audio file (.wav).
*   `--output-file <path>`: Optional. Path to save the converted female audio file (.wav). If omitted, it defaults to saving in `MODEL_SAVE_PATH` with `_converted` appended to the input filename.
*   `--model-checkpoint <path>`: Optional. Specify a particular `.h5` checkpoint file for the voice conversion model. If omitted, the script automatically uses the latest checkpoint found in `MODEL_SAVE_PATH`.
*   `--use-griffin-lim`: Optional. Force the use of the Griffin-Lim algorithm for audio synthesis instead of the HiFi-GAN vocoder. Useful if the vocoder is not trained or causing issues.

The script will:
1.  Load the specified (or latest) voice conversion model.
2.  Load the input audio file and preprocess it into a Mel spectrogram.
3.  Run the spectrogram through the conversion model.
4.  Attempt to load the trained HiFi-GAN vocoder (`hifigan_vocoder_generator.h5`).
5.  Synthesize the audio using the vocoder (or Griffin-Lim if specified or if the vocoder fails).
6.  Save the resulting audio file.

### 4. Combined Training (Default Behavior)

If you run the script without `--train-only`, `--train-vocoder-only`, or `--convert-only`, it will attempt to:
1.  Train the voice conversion model (or resume training).
2.  Train the HiFi-GAN vocoder (or load the existing one if `USE_PRETRAINED_VOCODER` is `True` and the file exists).

```bash
python your_script_name.py --epochs 50 --batch-size 8 --vocoder-epochs 100 --vocoder-batch-size 16
```

## Notes

*   Ensure the paths (`DATASET_PATH`, `MODEL_SAVE_PATH`) are correctly set in the script.
*   Training, especially vocoder training, can take a significant amount of time and computational resources (GPU recommended).
*   The quality of the conversion depends heavily on the dataset size, quality, model architecture, and training duration.
*   Monitor GPU memory usage; adjust `batch-size` if you encounter out-of-memory errors. Using the data generator (`USE_DATA_GENERATOR = True`, default) is highly recommended to prevent memory issues during conversion model training.
