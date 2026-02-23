import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

def visualize_comparison():
    # File paths
    normal_file = os.path.join("Actor_01", "03-01-02-01-01-01-01.wav")
    stressed_file = os.path.join("Actor_01", "03-01-05-01-01-01-01.wav")
    
    # Check if files exist
    if not os.path.exists(normal_file) or not os.path.exists(stressed_file):
        print("Error: Audio files not found.")
        return

    # Load audio
    y_normal, sr_normal = librosa.load(normal_file, sr=None)
    y_stressed, sr_stressed = librosa.load(stressed_file, sr=None)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Normal Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_normal, sr=sr_normal, color='blue')
    plt.title("Normal Speech Waveform (Calm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Stressed Waveform
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_stressed, sr=sr_stressed, color='red')
    plt.title("Stressed Speech Waveform (Angry)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    
    # Save output
    output_path = os.path.join("outputs", "waveform_comparison.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    visualize_comparison()
