import torch
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

class CustomLM(CTCDecoderLM):
    """Create a Python wrapper around `language_model` to feed to the decoder."""

    def __init__(self, language_model: torch.nn.Module):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.sil = -1  # index for silent token in the language model
        self.states = {}

        language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.language_model(self.sil)

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = self.language_model(token_index)
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)