import torch
import torchaudio
import os

bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H

def wer(actual_transcript,result):
    return torchaudio.functional.edit_distance(actual_transcript, result) / len(actual_transcript)

def read_file(file_path):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist or is not a file.")
    speech_file = file_path
    waveform, sample_rate = torchaudio.load(speech_file)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    return waveform

