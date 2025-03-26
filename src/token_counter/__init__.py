import sys
from loguru import logger
from typer import Typer
from tokenizers import Tokenizer

app = Typer(name="Token Counter")


def get_tokenizer(model: str = "gpt2") -> Tokenizer:
    return Tokenizer.from_pretrained(model)


def count_tokens_stream(tokenizer: Tokenizer, chunk_size: int = 1024 * 1024):
    """Process stdin in chunks and count tokens efficiently"""
    total_tokens = 0
    buffer = []

    while True:
        chunk = sys.stdin.read(chunk_size)
        if not chunk:
            break
        buffer.append(chunk)

        # Tokenize only the full text in buffer
        encoding = tokenizer.encode("".join(buffer))
        total_tokens += len(encoding.tokens)

        # Clear buffer to free memory
        buffer.clear()

    return total_tokens


@app.command()
def main(model: str = "gpt2"):
    """CLI tool to count tokens in text from stdin (handles large files)"""
    logger.info(f"Tokenizing text using: {model}")

    tokenizer = get_tokenizer(model)

    token_count = count_tokens_stream(tokenizer)

    logger.info(f"Token count: {token_count}")
