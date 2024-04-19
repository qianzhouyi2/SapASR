import torchaudio
from typing import List
from torchaudio.utils import download_asset
from decoder import *
from utils import *
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
acoustic_model = bundle.get_model()
waveform = read_file("ctctest/test.wav")
actual_transcript = "i really was very much afraid of showing him how much shocked i was at some parts of what he said"
actual_transcript = actual_transcript.split()
emission, _ = acoustic_model(waveform)
greedy_infer(emission,actual_transcript)
beamsearch_infer(emission,actual_transcript)
beamsearch_incremental_infer(emission,actual_transcript)
plot_alignments(waveform[0], emission, bundle.sample_rate)
beamsearch_nbest(emission,3)
common_params = {
    'word_score': WORD_SCORE,
}
search_params = {
    'beam size': [1, 5, 50, 500],
    'beam size token': [1, 5, 10, len(tokens)],
    'beam threshold': [1, 5, 10, 25],
    'lm weight': [0, LM_WEIGHT, 15]
}
for param, values in search_params.items():
    for value in values:
        decoder = kwarg_decoder(**common_params, **{param.replace(' ', '_'): value})
        print_decoded(decoder, emission, param, value)