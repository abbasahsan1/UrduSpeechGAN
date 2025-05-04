# UrduSER: A Comprehensive Dataset for Speech Emotion Recognition in Urdu Language

## Dataset Overview
The UrduSER (Urdu Speech Emotion Recognition) dataset is a comprehensive collection of audio recordings designed for speech emotion recognition research in the Urdu language. The dataset consists of emotional speech samples recorded by multiple actors, both male and female, expressing various emotional states.

## Dataset Structure
The dataset is organized into emotion-based directories, with each directory containing audio files corresponding to that particular emotion. The main emotion categories are:

1. **Angry (غصہ)** - Expressions of anger and irritation
2. **Fear (خوف)** - Expressions of fear and anxiety
3. **Boredom (بوریت)** - Expressions of boredom and disinterest
4. **Disgust (ناگواری)** - Expressions of disgust and repulsion
5. **Happy (خوشی)** - Expressions of happiness and joy
6. **Neutral (معمولی)** - Neutral expressions without strong emotion
7. **Sad (افسردہ)** - Expressions of sadness and sorrow

## Audio File Characteristics
All audio files in the dataset share the following technical specifications:
- **Format**: WAV (Waveform Audio File Format)
- **Sample Rate**: 41100 Hz
- **Bit Depth**: 32-Bit Float
- **Duration**: Varies from 1 to 17 seconds (most files are between 1-7 seconds)

## File Naming Convention
The audio files follow a specific naming convention that encodes important metadata:

`[Actor_ID]_[Gender_Code]_[Emotion_Code]_[Sequence_Number].wav`

Where:
- **Actor_ID**: Numeric identifier for the actor/speaker (ranges from 1-9 in the observed data)
- **Gender_Code**: 0 for male (مرد), 1 for female (عورت)
- **Emotion_Code**: Numeric code representing the emotion category
  - 1: Angry (غصہ)
  - 2: Fear (خوف)
  - 3: Boredom (بوریت)
  - 4: Disgust (ناگواری)
  - 5: Happy (خوشی)
  - 6: Neutral (معمولی)
  - 7: Sad (افسردہ)
- **Sequence_Number**: Two-digit sequence number (01, 02, 03, etc.) to differentiate between multiple recordings by the same actor for the same emotion

For example, the file `1_0_1_01.wav` represents:
- Actor ID: 1 (محمود اسلم)
- Gender: 0 (Male/مرد)
- Emotion: 1 (Angry/غصہ)
- Sequence: 01 (First recording in this category)

## Metadata Information
The dataset includes a comprehensive CSV file (UrSEC_Description.csv) that contains detailed metadata for each audio file, including:

1. **تشریح (Description)**: The transcription of the spoken text in Urdu
2. **جذبات (Emotion)**: The emotion category in Urdu (غصہ, خوف, بوریت, ناگواری, خوشی, معمولی, افسردہ)
3. **صنف (Gender)**: The gender of the speaker (مرد for male, عورت for female)
4. **اداکار (Actor)**: Name of the actor/speaker
5. **اداکار کاآئی ڈی نمبر (Actor ID)**: Numeric identifier for the actor
6. **نمونہ کی شکل (Sample Format)**: Audio format specification (32-Bit Float)
7. **نمونہ کی شرح (Sample Rate)**: Audio sampling rate (41100 Hz)
8. **فارمیٹ (Format)**: File format (WAV)
9. **دورانیہ سیکنڈز میں (Duration in Seconds)**: Length of the audio clip
10. **فائل کا نام (File Name)**: Name of the audio file
11. **سیریل نمبر (Serial Number)**: Sequential identifier for the entire dataset

## Dataset Statistics
The dataset contains thousands of audio samples (over 3,500 based on the CSV file) distributed across the seven emotion categories. The recordings feature multiple actors, providing diversity in vocal characteristics, speaking styles, and emotional expressions.

## Actors and Demographics
The dataset includes recordings from multiple actors, both male and female. Each actor is assigned a unique ID number, and their contributions are tracked in the metadata. Notable actors include:
- محمود اسلم (Mahmood Aslam) - Male, Actor ID: 1
- اسماء عباس (Asma Abbas) - Female, Actor ID: 9

## Potential Applications
This dataset is particularly valuable for:
1. **Speech Emotion Recognition (SER)** research specific to the Urdu language
2. **Cross-lingual emotion recognition** studies
3. **Affective computing** applications for Urdu-speaking populations
4. **Natural Language Processing (NLP)** research focused on emotional content in Urdu
5. **Human-Computer Interaction (HCI)** systems designed for Urdu speakers

## Dataset Uniqueness
The UrduSER dataset represents a significant contribution to speech emotion recognition research, as it addresses the lack of comprehensive emotional speech resources for the Urdu language. With its diverse emotional categories, multiple speakers, and detailed metadata, it provides researchers with a valuable tool for developing and evaluating speech emotion recognition systems specifically tailored for Urdu, one of the world's major languages with hundreds of millions of speakers.