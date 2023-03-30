"""Translate datasets using the inference server API."""

import os
import time
from pathlib import Path
from typing import Dict, List

import requests

from datasets import DatasetDict
from instructmultilingual.flores_200 import lang_name_to_code


def translation_request(
    url: str,
    source_language: str,
    target_language: str,
    texts: str,
) -> str:
    """Creates a HTTP POST request to the translation server.

    Args:
        url (str): URL of the server.
        source_language (str): Languague of the original text.
        target_language (str): Languague of the translated text.
        texts (str): the

    Returns:
        str: The translated text from the API
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "source_language": source_language,
        "target_language": target_language,
        "texts": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["translated_texts"]


def translate(
    example: Dict[str, str],
    url: str,
    source_lang_code: str,
    target_lang_code: str,
    keys_to_be_translated: List[str],
) -> Dict[str, str]:
    """Takes an example dictionary of translation keys and text values, and
    iterates over them to make translation requests to the inference API
    server.

    Args:
        example (Dict[str, str]): a dictionary of translation keys and text values to be translated.
        url (str): URL of the server.
        source_lang_code (str): Languague of the original text.
        target_lang_code (str): Languague of the translated text.
        keys_to_be_translated (List[str]): keys from example that should be translated via translation_request.

    Returns:
        Dict[str, str]: Translated example.
    """
    for key in keys_to_be_translated:
        example[key] = translation_request(url, source_lang_code, target_lang_code, example[key])
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
    num_proc: int = 8,
) -> None:
    """This function takes an DatasetDict object and translates it via the
    translation inference server API. The function then ouputs the translations
    in both json and csv formats into a output directory under the following
    naming convention:

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
            lambda x: translate(
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
            ))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
