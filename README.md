# WD Llava Caption

This repo is for an experiment I have been doing with llava 1.6 where I put descriptive tags in to the prompt and it seems to yield much better and accurate results about an image.

## Installation

1. Clone the repository and navigate to the directory:

    ```bash
    git clone https://github.com/ausboss/wd-llava-caption.git
    cd wd-llava-caption-main
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the server:

    ```bash
    python server.py
    ```

2. Open the Jupyter notebook:

    ```bash
    jupyter notebook notebook.ipynb
    ```

## Environment Variables

Set the environment variables as needed using `sample.env` as a template. You will need a Hugging Face token to download the model for generating tags. You can get a token by logging in and generating one [here](https://huggingface.co/settings/tokens).

## Additional Requirements

1. Install Ollama:

    ```bash
    Follow the instructions at https://ollama.com/
    ```

2. Pull the LLAVA model that fits your VRAM requirements:

    ```bash
    ollama pull <model name>
    ```

    - llava1.6 34b: 20GB
    - llava1.6 13b: 8GB

    More LLAVA 1.6 models can be found [here](https://ollama.com/library/llava:latest).


After installing Ollama and pulling the model update the `ollama_model` variable in `server.py` to match what you pulled.

## Files

- `server.py`: Start the server.
- `notebook.ipynb`: Jupyter notebook for image captioning.
- `requirements.txt`: Dependencies list.
- `sample.env`: Environment variables template.
