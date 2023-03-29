import os
import requests
import time

from datasets import DatasetDict
from instructmultilingual.flores_200 import lang_name_to_code
from multiprocessing import cpu_count
from pathlib import Path
from typing import List


def translate(url, source_language, target_language, texts):
    headers = {"Content-Type": "application/json"}
    data = {
        "source_language": source_language,
        "target_language": target_language,
        "texts": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["translated_texts"]


def tokenization(
    example,
    url,
    source_lang_code,
    target_lang_code,
    keys_to_be_translated=["dialogue", "summary"],
):
    for key in keys_to_be_translated:
        example[key] = translate(url, source_lang_code, target_lang_code, example[key])
    return example


def translate_dataset_via_api(
    dataset: DatasetDict,
    dataset_name: str,
    splits: List[str],
    translate_keys: List[str],
    target_language: str,
    url: str = "http://localhost:8000/translate",
    output_dir: str = "./datasets",
    source_language: str = "English",
    checkpoint: str = "facebook/nllb-200-3.3B",
    num_proc: int = cpu_count(),
) -> None:
    """This function takes an DatasetDict object and translates it via the translation inference server API. 
       The function then ouputs the translations in both json and csv formats into a output directory under the following naming convention:
       <root>/<dataset_name>/<target_language_code>/

    Args:
        dataset (DatasetDict): A DatasetDict object of the original text dataset. Needs to have at least one split.
        dataset_name (str): Name of the dataset for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        target_language (str): the language you want translation to.
        url (str, optional): The URL of the inference API server. Defaults to "http://localhost:8000/translate".
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to cpu_count().
    """

    source_language_code = lang_name_to_code[source_language]
    target_language_code = lang_name_to_code[target_language]

    checkpoint_str = checkpoint.replace("/", "-")
    translated_dir = Path(os.path.join(output_dir, dataset_name, target_language_code))
    translated_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for split in splits:
        split_time = time.time()
        ds = dataset[split]
        print(f"[{split}] {len(ds)=}")
        ds = ds.map(
            lambda x: tokenization(
                x,
                url=url,
                source_lang_code=source_language_code,
                target_lang_code=target_language_code,
                keys_to_be_translated=translate_keys,
            ),
            batched=True,
            num_proc=num_proc,
        )
        print(f"[{split}] One example translated {ds[0]=}")
        print(f"[{split}] took {time.time() - split_time:.4f} seconds")

        ds.to_csv(
            os.path.join(
                translated_dir,
                f"{dataset_name}_{split}_{target_language_code}_{checkpoint_str}.csv",
            ),
            index=False,
        )
        ds.to_json(
            os.path.join(
                translated_dir,
                f"{dataset_name}_{split}_{target_language_code}_{checkpoint_str}.jsonl",
            )
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
