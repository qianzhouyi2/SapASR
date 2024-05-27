import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder,download_pretrained_files
from typing import List
from utils import wer, read_file
import matplotlib.pyplot as plt
import time

LM_WEIGHT = 3.23
WORD_SCORE = -0.26
files = download_pretrained_files("librispeech-4-gram")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
tokens = [label.lower() for label in bundle.get_labels()]

beam_search_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=files.tokens,
    lm=files.lm,
    nbest=3,
    beam_size=1500,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    
greedy_decoder = GreedyCTCDecoder(tokens)

    
def greedy_infer(emission,actual_transcript):
    start_time = time.monotonic()
    greedy_result = greedy_decoder(emission[0])
    decode_time = time.monotonic() - start_time
    greedy_transcript = " ".join(greedy_result)
    greedy_wer = wer(actual_transcript, greedy_result)
    print(f"{decode_time:.4f} secs)")
    print(f"Transcript: {greedy_transcript}")
    print(f"WER: {greedy_wer}")

def beamsearch_infer(emission,actual_transcript):
    beam_search_result = beam_search_decoder(emission)
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    beam_search_wer = wer(actual_transcript, beam_search_result[0][0].words)

    print(f"Transcript: {beam_search_transcript}")
    print(f"WER: {beam_search_wer}")

def beamsearch_incremental_infer(emission,actual_transcript):
    beam_search_decoder.decode_begin()
    for t in range(emission.size(1)):
        beam_search_decoder.decode_step(emission[0, t:t + 1, :])
    beam_search_decoder.decode_end()
    beam_search_result_inc = beam_search_decoder.get_final_hypothesis()
    beam_search_transcript_inc = " ".join(beam_search_result_inc[0].words).strip()
    beam_search_wer_inc = wer(actual_transcript, beam_search_result_inc[0].words)
    print(f"Transcript: {beam_search_transcript_inc}")
    print(f"WER: {beam_search_wer_inc}")

def plot_alignments(waveform, emission, sample_rate):

    t = torch.arange(waveform.size(0)) / sample_rate
    ratio = waveform.size(0) / emission.size(1) / sample_rate

    chars = []
    words = []
    word_start = None
    beam_search_result = beam_search_decoder(emission)
    timesteps = beam_search_result[0][0].timesteps
    tokens = beam_search_decoder.idxs_to_tokens(beam_search_result[0][0].tokens)
    for token, timestep in zip(tokens, timesteps * ratio):
        if token == "|":
            if word_start is not None:
                words.append((word_start, timestep))
            word_start = None
        else:
            chars.append((token, timestep))
            if word_start is None:
                word_start = timestep

    fig, axes = plt.subplots(3, 1)

    def _plot(ax, xlim):
        ax.plot(t, waveform)
        for token, timestep in chars:
            ax.annotate(token.upper(), (timestep, 0.5))
        for word_start, word_end in words:
            ax.axvspan(word_start, word_end, alpha=0.1, color="red")
        ax.set_ylim(-0.6, 0.7)
        ax.set_yticks([0])
        ax.grid(True, axis="y")
        ax.set_xlim(xlim)

    _plot(axes[0], (1.0, 2.5))
    _plot(axes[1], (2.5, 4.0))
    _plot(axes[2], (4.0, 5.5))
    axes[2].set_xlabel("time (sec)")
    fig.tight_layout()
    plt.show()

def beamsearch_nbest(emission,n):
    beam_search_result = beam_search_decoder(emission)
    for i in range(n):
        transcript = " ".join(beam_search_result[0][i].words).strip()
        score = beam_search_result[0][i].score
        print(f"{transcript} (score: {score})")
