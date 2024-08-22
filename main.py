import torch
import torchaudio
from whisper_medusa import WhisperMedusaModel
from transformers import WhisperProcessor

model_name = "aiola/whisper-medusa-v1"
model = WhisperMedusaModel.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

path_to_audio = "../2024-08-22 11-19-24.wav"
SAMPLING_RATE = 16000
CHUNK_DURATION = 29  # seconds
language = "en"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_chunk(chunk):
    input_features = processor(
        chunk.squeeze(), return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features
    input_features = input_features.to(device)

    model_output = model.generate(
        input_features,
        language=language,
    )
    predict_ids = model_output[0]
    return processor.decode(predict_ids, skip_special_tokens=True)


# Load and preprocess audio
input_speech, sr = torchaudio.load(path_to_audio)
if input_speech.shape[0] > 1:  # If stereo, average the channels
    input_speech = input_speech.mean(dim=0, keepdim=True)

if sr != SAMPLING_RATE:
    input_speech = torchaudio.transforms.Resample(
        sr, SAMPLING_RATE)(input_speech)

# Calculate chunk size in samples
chunk_size = CHUNK_DURATION * SAMPLING_RATE

# Process audio in chunks
transcriptions = []
for i in range(0, input_speech.shape[1], chunk_size):
    chunk = input_speech[:, i:i+chunk_size]
    transcription = process_chunk(chunk)
    transcriptions.append(transcription)

# Combine transcriptions
full_transcription = " ".join(transcriptions)

print(full_transcription)

# Save output to file
with open("output.txt", "w") as f:
    f.write(full_transcription)
