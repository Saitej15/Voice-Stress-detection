import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tempfile

# Page configuration
st.set_page_config(page_title="Voice Stress Detector", layout="wide")

# ===============================
# 1️⃣ FEATURE EXTRACTION FUNCTION
# ===============================

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # STFT (Framing + Windowing)
    stft = np.abs(librosa.stft(y))

    # MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    # Pitch (Autocorrelation approximation)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])
    if np.isnan(pitch):
        pitch = 0

    features = np.hstack([
        mfcc_mean,
        zcr,
        spectral_centroid,
        energy,
        pitch
    ])

    return features, y, sr

# ===============================
# 2️⃣ UI LAYOUT
# ===============================

st.title("🎙️ Voice Stress Detection Dashboard")
st.markdown("""
Upload an audio file (.wav) to analyze the emotional stress level based on acoustic features.
The model predicts whether the speech appears **Normal** or **Stressed**.
""")

# Sidebar for model status
st.sidebar.title("Model Information")
if os.path.exists("model.joblib"):
    st.sidebar.success("✅ Model loaded successfully")
    model = joblib.load("model.joblib")
else:
    st.sidebar.error("❌ Model file (model.joblib) not found!")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Analyzing audio..."):
            # Extract features
            features, y_audio, sr = extract_features(tmp_path)
            
            # Make prediction
            prediction = model.predict(features.reshape(1, -1))[0]
            probability = model.predict_proba(features.reshape(1, -1))[0]
            
            # Display results
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == "normal":
                    st.success(f"### Prediction: NORMAL")
                else:
                    st.error(f"### Prediction: STRESSED")
                
                # Confidence score
                conf_idx = 0 if prediction == "normal" else 1
                st.metric("Confidence Score", f"{probability.max()*100:.2f}%")

            with col2:
                st.write("### Acoustic Highlights")
                st.write(f"- **Avg Pitch:** {features[-1]:.2f} Hz")
                st.write(f"- **Zero Crossing Rate:** {features[13]:.4f}")
                st.write(f"- **RMS Energy:** {features[15]:.4f}")

            # Visualizations
            st.divider()
            st.subheader("Visualizations")
            
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.write("**Waveform**")
                fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y_audio, sr=sr, ax=ax_wave)
                ax_wave.set_title("Audio Waveform")
                st.pyplot(fig_wave)

            with vis_col2:
                st.write("**Spectrogram**")
                fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax_spec)
                plt.colorbar(img, ax=ax_spec, format="%+2.f dB")
                ax_spec.set_title("Log-frequency Power Spectrogram")
                st.pyplot(fig_spec)

            st.write("**MFCC Time-Series**")
            fig_mfcc, ax_mfcc = plt.subplots(figsize=(12, 4))
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            img_mfcc = librosa.display.specshow(mfcc, x_axis='time', ax=ax_mfcc)
            plt.colorbar(img_mfcc, ax=ax_mfcc)
            ax_mfcc.set_title("MFCC")
            st.pyplot(fig_mfcc)

    except Exception as e:
        st.error(f"Error processing audio: {e}")
    finally:
        # Cleanup temporary file
        os.remove(tmp_path)
else:
    st.info("Upload a WAV file to begin analysis.")
