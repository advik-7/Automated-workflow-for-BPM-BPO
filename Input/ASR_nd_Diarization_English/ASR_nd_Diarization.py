# !pip install -q git+https://github.com/openai/whisper.git > /dev/null
# !pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null


import os
import datetime
import subprocess
import whisper
import torch
import numpy as np
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import wave
import contextlib


import os

def transcribe_and_diarize(mp3_path, num_speakers=2, model_size="base", output_file="transcription.txt"):
    """
    Transcribes and diarizes an audio file with multiple speakers.

    Parameters:
        mp3_path (str): Path to the MP3 file.
        num_speakers (int): Number of speakers in the audio.
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        output_file (str): Name of the output file to save the transcription.

    Returns:
        str: Path to the saved transcription file.
    """
    if not mp3_path.endswith('.wav'):
        wav_path = 'temp_audio.wav'
        try:
            subprocess.call(['ffmpeg', '-i', mp3_path, wav_path, '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(wav_path):
                raise FileNotFoundError("Conversion to WAV failed. Ensure the input file is valid.")
        except Exception as e:
            raise RuntimeError(f"Failed to convert MP3 to WAV: {e}")
    else:
        wav_path = mp3_path

    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(wav_path)
        segments = result["segments"]

        with contextlib.closing(wave.open(wav_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        audio = Audio()
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda"))
        embeddings = np.zeros(shape=(len(segments), 192))

        for i, segment in enumerate(segments):
            start = segment["start"]
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(wav_path, clip)
            embeddings[i] = embedding_model(waveform[None])

        embeddings = np.nan_to_num(embeddings)
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_

        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        def time(secs):
            return datetime.timedelta(seconds=round(secs))

        with open(output_file, "w") as f:
            for i, segment in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    f.write(f"\n{segment['speaker']} {time(segment['start'])}\n")
                f.write(segment["text"][1:] + " ")
                f.write("\n\n")  # Add extra spacing between speaker segments for better readability.

        if wav_path == 'temp_audio.wav':
            os.remove(wav_path)

        return f"Transcription saved to {output_file}"
    except Exception as e:
        if wav_path == 'temp_audio.wav' and os.path.exists(wav_path):
            os.remove(wav_path)
        raise RuntimeError(f"Error during transcription or diarization: {e}")




audio="/content/clipped_audio.mp3"
transcribe_and_diarize(audio)
