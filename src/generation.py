"""Answer generation with flan-t5 for the end-to-end RAG experiment."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config
from .prompting import build_prompt


class AnswerGenerator:
    def __init__(
        self,
        model_name=config.GENERATOR,
        device=None,
        max_input_tokens=config.MAX_INPUT_TOKENS,
        max_new_tokens=config.MAX_NEW_TOKENS,
        num_beams=config.NUM_BEAMS,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        # Never exceed the input length the model declares (512 tokens for flan-t5).
        model_limit = getattr(self.tokenizer, "model_max_length", max_input_tokens)
        self.max_input_tokens = min(max_input_tokens, model_limit)
        if self.max_input_tokens < max_input_tokens:
            print(f"max_input_tokens reduced from {max_input_tokens} to the model limit "
                  f"of {self.max_input_tokens}")
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.last_prompt_tokens = 0

    def build_prompt(self, query, docs):
        return build_prompt(self.tokenizer, query, docs, self.max_input_tokens)

    @torch.no_grad()
    def generate(self, query, docs):
        prompt, _ = self.build_prompt(query, docs)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens,
        ).to(self.device)
        self.last_prompt_tokens = int(inputs["input_ids"].shape[1])
        assert self.last_prompt_tokens <= self.max_input_tokens
        output = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, num_beams=self.num_beams,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
