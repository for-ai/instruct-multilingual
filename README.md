# Instruct Multilingual

## Setup

```shell
conda create -n instructmultilingual python=3.8.10 -y
conda activate instructmultilingual
pip install -r requirements.txt
```

## Dataset Projection

### [PromptSource](https://github.com/bigscience-workshop/promptsource)

```shell
DUMP_FOLDER='' # fill this with your desired address
SRC_DATA_FOLDER=$DUMP_FOLDER/projection_from_psrc
mkdir -p $SRC_DATA_FOLDER
mkdir -p $SRC_DATA_FOLDER/cache

python data/project_from_psrc.py \
--dataset-name-or-paths glue glue glue glue glue \
--dataset-configs cola sst2 mrpc qqp stsb \
--prompt-templates-configs None None None None None \
--cache-dir $SRC_DATA_FOLDER/cache \
--output-dir $SRC_DATA_FOLDER \
--highlight-variables \
--add-source-metadata \
--num-proc 16
```

See the details of the arguments by, 

```shell
python data/project_from_psrc.py --help
```

## Translate

```shell
python -m instructmultilingual.translate \
          --text="Cohere For AI will make the best instruct multilingual model in the world" \
          --source_language="English" \
          --target_language="Egyptian Arabic"
```
