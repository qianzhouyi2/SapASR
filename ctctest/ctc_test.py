import torchaudio
from typing import List
from torchaudio.utils import download_asset
from decoder import *
from utils import *
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
acoustic_model = bundle.get_model()
waveform = read_file("ctctest/test2.wav")
actual_transcript = "harbin institute of technology is one of the top universities in china"
actual_transcript = actual_transcript.split()
emission, _ = acoustic_model(waveform)
print(files)
greedy_infer(emission,actual_transcript)
beamsearch_infer(emission,actual_transcript)
beamsearch_incremental_infer(emission,actual_transcript)
plot_alignments(waveform[0], emission, bundle.sample_rate)
beamsearch_nbest(emission,3)
beam_sizes = [1, 5, 50, 500]

def print_decoded(decoder, emission, param, param_value):
    start_time = time.monotonic()
    result = decoder(emission)
    decode_time = time.monotonic() - start_time
    transcript = " ".join(result[0][0].words).lower().strip()
    score = result[0][0].score
    print(f"{param} {param_value:<3}: {transcript} (score: {score:.2f}; {decode_time:.4f} secs)")

for beam_size in beam_sizes:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        beam_size=beam_size,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam size", beam_size)

num_tokens = len(tokens)
beam_size_tokens = [1, 5, 10, num_tokens]

for beam_size_token in beam_size_tokens:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        beam_size_token=beam_size_token,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam size token", beam_size_token)

beam_thresholds = [1, 5, 10, 25]

for beam_threshold in beam_thresholds:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        beam_threshold=beam_threshold,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam threshold", beam_threshold)

lm_weights = [0, LM_WEIGHT, 15]

for lm_weight in lm_weights:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        lm_weight=lm_weight,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "lm weight", lm_weight)