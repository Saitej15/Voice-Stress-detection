# 🎙️ Voice Stress Detection Project

This project implements a machine learning-based system to detect emotional stress in human speech. It extracts key acoustic features from audio recordings and uses a trained model to classify speech as **Normal** or **Stressed**.

## 🚀 Features

- **Automated Feature Extraction**: Extracts MFCCs, Zero Crossing Rate, Spectral Centroid, RMS Energy, and Pitch.
- **Machine Learning Analysis**: Classifies voice samples using a pre-trained `joblib` model.
- **Interactive Dashboard**: A built-in Streamlit web application for real-time file analysis and visualization.
- **Detailed Visualizations**: Generates Waveforms, Spectrograms, and MFCC heatmaps.
- **Comprehensive Reporting**: Scripts to generate Word-based reports of the analysis results.

## 🛠️ Project Structure

- `streamlit_app.py`: The main interactive dashboard for users to upload and analyze WAV files.
- `voice_stress_detection.py`: Core logic for feature extraction and model inference.
- `visualize_waveforms.py`: Script for generating waveform plots.
- `visualize_mfcc_heatmap.py`: Script for generating MFCC visualizations.
- `model.joblib`: The serialized pre-trained machine learning model.
- `Voice_Stress_Detection_Case_Study.docx`: Detailed documentation and case study report.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Saitej15/Voice-Stress-detection.git
   cd Voice-Stress-detection
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit librosa numpy matplotlib joblib scikit-learn
   ```

## 🖥️ Usage

To run the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Upload any `.wav` file to see the stress prediction and acoustic analysis.

## 📊 Dataset Attribution

This project utilizes the **RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. 

> [!NOTE]
> The raw dataset files (audio folders like `Actor_01`, etc.) are excluded from this repository due to their large size. You can download the dataset from [Kaggle](https://www.kaggle.com/uwrfkistler/ravdess-emotional-speech-audio) or the [official RAVDESS page](https://zenodo.org/record/1188388).

## 📄 License

This project is for educational and research purposes.
