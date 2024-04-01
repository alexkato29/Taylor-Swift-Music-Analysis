from pathlib import Path

import pandas as pd

import torch
import torchaudio

# from transformers import AutoProcessor, AutoModelForAudioClassification
from transformers import Wav2Vec2Processor, Wav2Vec2Model


audio = pd.read_csv('data/audio.csv')

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def encode_audio(file_path, segment_length=30):
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert stereo to mono by averaging the two channels
    if waveform.shape[0] == 2:  # Check if the audio is stereo
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Calculate total samples for a k-second segment
    total_samples = segment_length * sample_rate
    
    if waveform.shape[1] > total_samples:
        midpoint = waveform.shape[1] // 2
        
        start_sample = max(0, midpoint - (total_samples // 2))
        end_sample = start_sample + total_samples
        
        if end_sample > waveform.shape[1]:
            end_sample = waveform.shape[1]
            start_sample = max(0, end_sample - total_samples)
        
        waveform = waveform[:, start_sample:end_sample]
    
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Ensure the waveform is in the correct shape for the processor
    inputs = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        embeddings = model(inputs).last_hidden_state
        print(file_path)
    # Aggregate embeddings if needed
    return embeddings.mean(dim=1).numpy()

# Process and encode each MP3
audio_embeddings = []
for track_name in audio['track_name']:
    mp3_path = f'data/mp3s/{track_name}.mp3'
    if Path(mp3_path).is_file():
        encoding = encode_audio(mp3_path)
        audio_embeddings.append(encoding)
    else:
        audio_embeddings.append(None)

# Update the DataFrame
audio['latent_audio'] = audio_embeddings
audio['latent_audio'] = audio['latent_audio'].apply(lambda x: x.flatten().tolist() if x is not None else None)

audio.to_csv("data/audio.csv", index=False)
