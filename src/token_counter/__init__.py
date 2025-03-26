import sys
from loguru import logger
from typer import Typer
from tokenizers import Tokenizer, Encoding

app = Typer(name="Token Counter")


def get_tokenizer(model: str = "gpt2") -> Tokenizer:
    return Tokenizer.from_pretrained(model)


def count_tokens_stream(tokenizer: Tokenizer, batch_size: int = 1000):
    """Stream stdin, tokenize in batches, and count tokens efficiently"""
    total_tokens = 0
    buffer = []

    for line in sys.stdin:
        buffer.append(line.strip())

        if len(buffer) >= batch_size:
            # Tokenize batch and count tokens
            encodings: list[Encoding] = tokenizer.encode_batch(buffer)
            total_tokens += sum(len(enc.tokens) for enc in encodings)
            buffer.clear()  # Free memory

    # Process remaining lines in buffer
    if buffer:
        encodings: list[Encoding] = tokenizer.encode_batch(buffer)
        total_tokens += sum(len(enc.tokens) for enc in encodings)

    return total_tokens


@app.command()
def main(model: str = "gpt2"):
    """CLI tool to count tokens in text from stdin efficiently"""
    logger.info(f"Tokenizing text using: {model}")

    tokenizer = get_tokenizer(model)
    token_count = count_tokens_stream(tokenizer)

    logger.info(f"Token count: {token_count}")
