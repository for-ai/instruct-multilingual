# Instruct Multilingual

## Setup

```shell
conda create -n instructmultilingual python=3.8.10 -y
conda activate instructmultilingual
pip install -r requirements.txt
```


## Inference Server

### Convert the models first

```shell
mkdir models
ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir models/nllb-200-3.3B-converted
```

### Run the server locally
```shell
uvicorn instructmultilingual.server:app --host 0.0.0.0 --port 8000
```

### Using docker to run the server
```shell
# Build
docker build -t instruct-multilingual .

# Run
docker run -it --rm --gpus 1,2,3,4,5,6,7,8 -p 8000:8000 -v $(pwd):/instruct-multilingual instruct-multilingual
```

### Client Side

This script translate the samsum dataset using the inference server
```
python main.py
```

## Translate

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
    num_proc= 8,
)
```

