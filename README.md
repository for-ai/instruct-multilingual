# Instruct Multilingual

## Setup

```shell
conda create -n instructmultilingual python=3.8.10 -y
conda activate instructmultilingual
pip install -r requirements.txt
```

## Translate

```shell
python -m instructmultilingual.translate \
          --text="Cohere For AI will make the best instruct multilingual model in the world" \
          --source_language="English" \
          --target_language="Egyptian Arabic"
```
