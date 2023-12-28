import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

input_folder = "W:\\Data\\ADD\\track1"
output_folder_fake = "W:\\workdir3\\test\\fake"
output_folder_real = "W:\\workdir3\\test\\real"

def pad(x, max_len=48000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

if not os.path.exists(output_folder_fake):
    os.makedirs(output_folder_fake)

if not os.path.exists(output_folder_real):
    os.makedirs(output_folder_real)

# Parameters for mel spectrogram calculation
hop_length = 512
factor = 1
sr = 16000

# Function to save mel spectrogram as PNG image
def save_mel_spectrogram(input_file, label):
    print(f"Processing file: {input_file}")

    signal, sr = librosa.load(input_file)
    signal = pad(signal)

    audiolength = librosa.get_duration(y=signal, sr=sr, hop_length=hop_length)
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length)
    spectrogram = librosa.power_to_db(mel_signal, ref=np.max)

    plt.figure(figsize=(factor * audiolength, 1))
    plt.axis('off')
    plt.imshow(spectrogram, cmap='magma', aspect='auto', extent=[0, 1, 0, 1])

    output_folder = output_folder_fake if label == 'fake' else output_folder_real
    output_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}.png"
    output_path = os.path.join(output_folder, output_filename)

    print(f"Saving as {label}: {output_path}")
    plt.savefig(output_path, dpi=224, bbox_inches="tight", pad_inches=0)
    plt.close()

# Process each audio file in the input folder
label_file = "W:\\Data\\ADD\\label\\label1.txt"
with open(label_file, 'r') as labels:
    label_data = labels.readlines()

# Process each audio file in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            input_file = os.path.join(root, file)

            filename = os.path.splitext(os.path.basename(input_file))[0]
            for line in label_data:
                file_name, label = line.strip().split()
                if file_name == f"{filename}.wav":
                    print(f"Match found: {file_name}, {label}")
                    save_mel_spectrogram(input_file, label)
                    break  # Stop searching for labels once found