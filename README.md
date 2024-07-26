# Instruct Multilingual
This repository contains code to translate datasets into multiple languages using the NLLB (No Language Left Behind) model. The model can be used:
- [From a server to translate a single text or entire datasets.](##Inference-Server)
- [Using the `translate` script to translate a single text.](##Translate)

Note: This repository has been tested on a Linux machine using a Nvidia GPU. The code assumes access to a GPU. Depending on your hardware, you might need to modify the code to change the number of GPUs and the batch size.


## Setup
It is recommended to use a virtual environment to install the dependencies.

```shell
conda create -n instructmultilingual python=3.8.10 -y
conda activate instructmultilingual
pip install -r requirements.txt
```

## Inference Server
The inference server is a FastAPI application that can be used to translate a single text or entire datasets.

### Convert the models first
For efficient inference, the model is converted using [CTranslate2](https://github.com/OpenNMT/CTranslate2).
```shell
mkdir models
ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir models/nllb-200-3.3B-converted
```

### Run the server locally
To start the server, we need to run the following command:
```shell
uvicorn instructmultilingual.server:app --host 0.0.0.0 --port 8000
```

### Using docker to run the server
To run the server using docker, we need to build and run the docker image, and run the server.
```shell
# Build
docker build -t instruct-multilingual .

# Run
docker run -it --rm --gpus 1 -p 8000:8000 -v $(pwd):/instruct-multilingual instruct-multilingual
```

### Client Side
This script translate the [samsum](https://huggingface.co/datasets/samsum) dataset using the inference server. It showcases how to use the inference server to translate a single text and entire datasets.
```
python main.py
```

## Translate
We also provide a script to translate a single text from the CLI. This script downloads the model from Hugging Face and translates the text provided into the target language.
```shell
python -m instructmultilingual.translate \
          --text="Cohere For AI will make the best instruct multilingual model in the world" \
          --source_language="English" \
          --target_language="Egyptian Arabic"
```

## Translate an instructional dataset from xP3 (or any dataset repo from HuggingFace Hub)
An example of using `translate_dataset_from_huggingface_hub` to translate PIQA with the `finish_sentence_with_correct_choice` template into languages used by Multilingual T5 (mT5) model

```python
from instructmultilingual.translate_datasets import translate_dataset_from_huggingface_hub

translate_dataset_from_huggingface_hub(
    repo_id = "bigscience/xP3",
    train_set = ["en/xp3_piqa_None_train_finish_sentence_with_correct_choice.jsonl"],
    validation_set = ["en/xp3_piqa_None_validation_finish_sentence_with_correct_choice.jsonl"],
    test_set = [],
    dataset_name="PIQA",
    template_name="finish_sentence_with_correct_choice",
    splits=["train", "validation"],
    translate_keys=["inputs", "targets"],
    url= "http://localhost:8000/translate",
    output_dir= "/home/weiyi/instruct-multilingual/datasets",
    source_language= "English",
    checkpoint="facebook/nllb-200-3.3B",
)
```

