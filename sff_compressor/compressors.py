from .methods.selective_context_compressor import SCCompressor
from .methods.kis import KiSCompressor
from .methods.scrl_compressor import SCRLCompressor
from .methods.llmlingua_compressor_pro import LLMLinguaCompressor
import tiktoken
from typing import List, Dict


class PromptCompressor:
    def __init__(self, type: str = 'SCCompressor', lang: str = 'en', model='gpt2', device='cuda', model_dir: str = '',
                 use_auth_token: bool = False, open_api_config: dict = {}, token: str = '',
                 tokenizer_dir: str = "sentence-transformers/paraphrase-distilroberta-base-v2"):
        self.type = type
        if self.type == 'SCCompressor':
            self.compressor = SCCompressor(lang=lang, model=model, device=device)
        elif self.type == 'KiSCompressor':
            self.compressor = KiSCompressor(DEVICE=device, model_dir=model_dir)
        elif self.type == 'LLMLinguaCompressor':
            self.compressor = LLMLinguaCompressor(device_map=device, model_name=model_dir,
                                                  use_auth_token=use_auth_token, open_api_config=open_api_config,
                                                  token=token)
        elif self.type == 'LongLLMLinguaCompressor':
            self.compressor = LLMLinguaCompressor(device_map=device, model_name=model_dir,
                                                  use_auth_token=use_auth_token, open_api_config=open_api_config,
                                                  token=token)
        elif self.type == 'LLMLingua2Compressor':
            self.compressor = LLMLinguaCompressor(device_map=device, model_name=model_dir,
                                                  use_auth_token=use_auth_token, open_api_config=open_api_config,
                                                  token=token,
                                                  use_llmlingua2=True)
        elif self.type == 'SCRLCompressor':
            if model_dir:
                self.compressor = SCRLCompressor(model_dir=model_dir, device=device, tokenizer_dir=tokenizer_dir)
            else:
                print("model_dir parameter is required")

    def compressgo(self, original_prompt: str = '', ratio: float = 0.5, level: str = 'phrase',
                   max_length: int = 256, num_beams: int = 4, do_sample: bool = True, num_return_sequences: int = 1,
                   target_index: int = 0, instruction: str = "", question: str = "", target_token: float = -1,
                   iterative_size: int = 200, force_context_ids: List[int] = None, force_context_number: int = None,
                   use_sentence_level_filter: bool = False, use_context_level_filter: bool = True,
                   use_token_level_filter: bool = True, keep_split: bool = False, keep_first_sentence: int = 0,
                   keep_last_sentence: int = 0, keep_sentence_number: int = 0, high_priority_bonus: int = 100,
                   context_budget: str = "+100", token_budget_ratio: float = 1.4, condition_in_question: str = "none",
                   reorder_context: str = "original", dynamic_context_compression_ratio: float = 0.0,
                   condition_compare: bool = False, add_instruction: bool = False, rank_method: str = "llmlingua",
                   concate_question: bool = True):
        if self.type == 'SCCompressor':
            compressed = self.compressor.compress(original_prompt=original_prompt, ratio=ratio, level=level)
        elif self.type == 'KiSCompressor':
            compressed = self.compressor.compress(original_prompt=original_prompt, ratio=ratio, max_length=max_length,
                                                  num_beams=num_beams, do_sample=do_sample,
                                                  num_return_sequences=num_return_sequences, target_index=target_index)
        elif self.type == 'SCRLCompressor':
            compressed = self.compressor.compress(original_prompt=original_prompt, ratio=ratio, max_length=max_length)
        elif self.type in ['LLMLinguaCompressor', 'LongLLMLinguaCompressor', 'LLMLingua2Compressor']:
            compressed = self.compressor.compress(context=original_prompt, ratio=ratio, instruction=instruction,
                                                  question=question, target_token=target_token,
                                                  iterative_size=iterative_size, force_context_ids=force_context_ids,
                                                  force_context_number=force_context_number,
                                                  use_token_level_filter=use_token_level_filter,
                                                  use_context_level_filter=use_context_level_filter,
                                                  use_sentence_level_filter=use_sentence_level_filter,
                                                  keep_split=keep_split, keep_first_sentence=keep_first_sentence,
                                                  keep_last_sentence=keep_last_sentence,
                                                  keep_sentence_number=keep_sentence_number,
                                                  high_priority_bonus=high_priority_bonus,
                                                  context_budget=context_budget, token_budget_ratio=token_budget_ratio,
                                                  condition_in_question=condition_in_question,
                                                  reorder_context=reorder_context,
                                                  dynamic_context_compression_ratio=dynamic_context_compression_ratio,
                                                  condition_compare=condition_compare,
                                                  add_instruction=add_instruction, rank_method=rank_method,
                                                  concate_question=concate_question)
        else:
            compressed = self.compressor.compress(original_prompt=original_prompt, ratio=ratio)

        # Extract compressed text from the returned dictionary
        compressed_text = compressed.get('compressed_prompt', '') if isinstance(compressed, dict) else compressed

        # Ensure formatting is preserved with newline replaced by "\n"
        return self._preserve_formatting(original_prompt, compressed_text)

    @staticmethod
    def _preserve_formatting(original_prompt: str, compressed_prompt: str) -> str:
        """
        Restores the formatting (newlines, special characters, etc.) from the original prompt to the compressed prompt,
        replacing actual newlines with the string "\n".
        """
        formatted_prompt = original_prompt
        if '\n' in original_prompt:
            # If there are newlines in the original prompt, replace them with "\n"
            formatted_prompt = compressed_prompt.replace('\n', '\\n')
        return formatted_prompt

    def count_tokens_with_tiktoken(self, text: str) -> int:
        # Use the appropriate encoding for the GPT model (e.g., 'cl100k_base' for GPT-3.5 and GPT-4)
        encoder = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text and return the number of tokens
        tokens = encoder.encode(text)
        return len(tokens)

    def compress_and_count(self, original_prompt: str, ratio: float = 0.5) -> Dict[str, int]:
        compressed_text = self.compressgo(original_prompt=original_prompt, ratio=ratio)

        # Calculate token counts using tiktoken
        original_tokens = self.count_tokens_with_tiktoken(original_prompt)
        compressed_tokens = self.count_tokens_with_tiktoken(compressed_text)

        return {
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'compressed_prompt': compressed_text
        }
