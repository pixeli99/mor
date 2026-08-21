import os

from transformers import AutoTokenizer

from paths import MODEL_DIR

# Tokenizer files are vendored under MODEL_DIR (no hub access on cluster nodes).
TOKENIZERS = {
    "smollm": AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "SmolLM-135M")),
    "smollm2": AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "SmolLM2-135M")),
    "dsv4": AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "DeepSeek-V4-Tokenizer")),
}


def load_tokenizer_from_config(cfg):
    tokenizer = TOKENIZERS[cfg.tokenizer]
    if tokenizer.pad_token is None:
        if cfg.tokenizer in ["smollm", "smollm2"]:
            # '<|endoftext|>'
            tokenizer.pad_token_id = 0
        else:
            raise ValueError(f"Tokenizer {cfg.tokenizer} does not have a pad token, please specify one in the config")
    return tokenizer