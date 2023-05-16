# ggify

A small tool that downloads models from [the Huggingface Hub](https://huggingface.co/models) and converts them into GGML for use with [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Usage

- Download and compile `llama.cpp`.
- Set up a virtualenv using the requirements from `llama.cpp`.
- Install the requirements from this repo in that virtualenv.
- Run e.g. `python ggify.py databricks/dolly-v2-12b` (nb.: I haven't tried with that particular repo)
- You'll end up with GGML models under `models/...`.
