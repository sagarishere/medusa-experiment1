import torch
import torchaudio
import os
from whisper_medusa import WhisperMedusaModel
from transformers import WhisperProcessor
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

model_name = "aiola/whisper-medusa-v1"
model = WhisperMedusaModel.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

path_to_audio = "../2024-08-22 11-19-24.wav"
SAMPLING_RATE = 16000
CHUNK_DURATION = 29  # seconds
language = "en"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory if it doesn't exist
output_dir = "./audiochunks"
os.makedirs(output_dir, exist_ok=True)


def process_chunk(chunk_path):
    print(f"{Fore.YELLOW}Processing chunk: {chunk_path}")
    chunk, sr = torchaudio.load(chunk_path)
    input_features = processor(
        chunk.squeeze(), return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features
    input_features = input_features.to(device)

    model_output = model.generate(
        input_features,
        language=language,
    )
    predict_ids = model_output[0]
    transcription = processor.decode(predict_ids, skip_special_tokens=True)
    print(f"{Fore.GREEN}Transcription complete for {chunk_path}")
    return transcription


# Load and preprocess audio
print(f"{Fore.CYAN}Loading audio file: {path_to_audio}")
input_speech, sr = torchaudio.load(path_to_audio)
if input_speech.shape[0] > 1:  # If stereo, average the channels
    input_speech = input_speech.mean(dim=0, keepdim=True)

if sr != SAMPLING_RATE:
    input_speech = torchaudio.transforms.Resample(
        sr, SAMPLING_RATE)(input_speech)

# Calculate chunk size in samples
chunk_size = CHUNK_DURATION * SAMPLING_RATE

# Process audio in chunks
print(f"{Fore.CYAN}Starting chunk processing")
chunk_count = 0
with open("output.txt", "w") as output_file:
    for i in range(0, input_speech.shape[1], chunk_size):
        chunk_count += 1
        chunk = input_speech[:, i:i+chunk_size]

        # Save chunk to file
        chunk_filename = f"{output_dir}/{chunk_count:05d}.wav"
        torchaudio.save(chunk_filename, chunk, SAMPLING_RATE)
        print(f"{Fore.BLUE}Saved chunk: {chunk_filename}")

        # Process the chunk
        transcription = process_chunk(chunk_filename)

        # Write transcription to file
        output_file.write(transcription + " ")
        output_file.flush()  # Ensure the content is written immediately

        print(
            f"{Fore.MAGENTA}Transcription added to output file for chunk {chunk_count}")

print(f"{Fore.GREEN}Processing complete. Total chunks processed: {chunk_count}")
print(f"{Style.BRIGHT}{Fore.CYAN}Full transcription saved in output.txt")
