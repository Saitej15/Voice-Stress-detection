import os
import zipfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===============================
# 1️⃣ DOWNLOAD DATASET FROM KAGGLE
# ===============================

print("Downloading RAVDESS Dataset from Kaggle...")

if not any(os.path.isdir(d) for d in os.listdir(".") if d.startswith("Actor_")):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Downloading Dataset...")
    api.dataset_download_files("uwrfkaggler/ravdess-emotional-speech-audio", path=".", unzip=True)

if not os.path.exists("outputs"):
    os.makedirs("outputs")

print("Dataset Ready!\n")

# ===============================
# 2️⃣ FEATURE EXTRACTION FUNCTION
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

    features = np.hstack([
        mfcc_mean,
        zcr,
        spectral_centroid,
        energy,
        pitch
    ])

    return features

# ===============================
# 3️⃣ PREPARE DATASET (Normal vs Stressed)
# ===============================

dataset_path = "."

data = []
labels = []

print("Extracting Features...")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # RAVDESS emotion code
            emotion_code = file.split("-")[2]

            # Calm = Normal
            if emotion_code == "02":
                label = "normal"

            # Angry (05) + Fearful (06) = Stressed
            elif emotion_code in ["05", "06"]:
                label = "stressed"

            else:
                continue

            features = extract_features(file_path)
            data.append(features)
            labels.append(label)

X = np.array(data)
y = np.array(labels)

print("Total Samples:", len(X))

# ===============================
# 4️⃣ TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5️⃣ TRAIN RANDOM FOREST MODEL
# ===============================

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "model.joblib")
print("\nModel saved to model.joblib")

# ===============================
# 6️⃣ CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=["Normal", "Stressed"],
            yticklabels=["Normal", "Stressed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
print("Confusion Matrix saved to outputs/confusion_matrix.png")
# plt.show()

# ===============================
# 7️⃣ TIME SERIES ANALYSIS
# ===============================

example_file = None

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            example_file = os.path.join(root, file)
            break
    if example_file:
        break

y_audio, sr = librosa.load(example_file, sr=None)

# Waveform
plt.figure()
librosa.display.waveshow(y_audio, sr=sr)
plt.title("Waveform")
plt.savefig("outputs/waveform.png")
print("Waveform saved to outputs/waveform.png")
# plt.show()

# Spectrogram
plt.figure()
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title("Spectrogram")
plt.savefig("outputs/spectrogram.png")
print("Spectrogram saved to outputs/spectrogram.png")
# plt.show()

# MFCC Time Series
mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)

plt.figure()
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC Time-Series")
plt.savefig("outputs/mfcc_timeseries.png")
print("MFCC Time-Series saved to outputs/mfcc_timeseries.png")
# plt.show()

# ===============================
# 8️⃣ VARIANCE COMPARISON
# ===============================

df = pd.DataFrame(X)
df['label'] = y

normal_var = df[df['label'] == 'normal'].iloc[:, :-1].var().mean()
stress_var = df[df['label'] == 'stressed'].iloc[:, :-1].var().mean()

print("\nAverage Feature Variance:")
print("Normal Speech Variance:", normal_var)
print("Stressed Speech Variance:", stress_var)

plt.figure()
sns.barplot(x=["Normal", "Stressed"],
            y=[normal_var, stress_var])
plt.title("Variance Comparison")
plt.savefig("outputs/variance_comparison.png")
print("Variance Comparison saved to outputs/variance_comparison.png")
# plt.show()