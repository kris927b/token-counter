import subprocess


def test_basic_token_count():
    # Test with a simple input
    input_text = "Hello world this is a test"
    process: subprocess.CompletedProcess[str] = subprocess.run(
        ["token-counter"], input=input_text, text=True, capture_output=True
    )
    assert process.returncode == 0
    assert "Token count: 6" in process.stderr
