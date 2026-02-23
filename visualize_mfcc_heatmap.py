import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

def visualize_mfcc_comparison():
    # File paths (Using the same representative samples from Actor_01)
    normal_file = os.path.join("Actor_01", "03-01-02-01-01-01-01.wav")
    stressed_file = os.path.join("Actor_01", "03-01-05-01-01-01-01.wav")
    
    # Check if files exist
    if not os.path.exists(normal_file) or not os.path.exists(stressed_file):
        print("Error: Audio files not found.")
        return

    # Load audio
    y_normal, sr_normal = librosa.load(normal_file, sr=None)
    y_stressed, sr_stressed = librosa.load(stressed_file, sr=None)

    # Extract MFCCs
    mfcc_normal = librosa.feature.mfcc(y=y_normal, sr=sr_normal, n_mfcc=13)
    mfcc_stressed = librosa.feature.mfcc(y=y_stressed, sr=sr_stressed, n_mfcc=13)

    # Create plot
    plt.figure(figsize=(12, 10))

    # Normal MFCC Heatmap
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc_normal, x_axis='time', sr=sr_normal)
    plt.colorbar(format='%+2.0f dB')
    plt.title("MFCC Heatmap - Normal Speech (Calm)")
    plt.ylabel("MFCC Coefficients")

    # Stressed MFCC Heatmap
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfcc_stressed, x_axis='time', sr=sr_stressed)
    plt.colorbar(format='%+2.0f dB')
    plt.title("MFCC Heatmap - Stressed Speech (Angry)")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()
    
    # Save output
    output_path = os.path.join("outputs", "mfcc_heatmap_comparison.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    print(f"MFCC Heatmap comparison saved to {output_path}")

if __name__ == "__main__":
    visualize_mfcc_comparison()
