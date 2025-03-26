import sys
from loguru import logger
from typer import Typer, echo
from tokenizers import Tokenizer, Encoding

app = Typer(name="Token Counter")


def get_tokenizer(model: str = "gpt2") -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_pretrained(model)
    return tokenizer


@app.command()
def main(model: str = "gpt2"):
    """CLI tool to count tokens in text from stdin"""
    # Read text from stdin
    text = sys.stdin.read().strip()

    if not text:
        logger.error("No input provided. Please pipe or redirect text to stdin.")
        sys.exit(1)

    logger.info(f"Tokenizing text using: {model}")

    tokenizer = get_tokenizer(model)

    # Tokenize the input text
    # Since we're not training, we'll just encode directly
    encoding: Encoding = tokenizer.encode(text)
    token_count = len(encoding.tokens)

    # Output results
    logger.info(f"Token count: {token_count}")
