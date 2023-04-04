import os
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import requests

from datasets import DatasetDict
from instructmultilingual.flores_200 import lang_name_to_code


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
        template_name: str,
        splits: List[str],
        translate_keys: List[str],
        target_language: str,
        url: str = "http://localhost:8000/translate",
        output_dir: str = "./datasets",
        source_language: str = "English",
        checkpoint: str = "facebook/nllb-200-3.3B",
        num_proc: int = cpu_count(),
) -> None:
    """This function takes an DatasetDict object and translates it via the
    translation inference server API. The function then ouputs the translations
    in both json and csv formats into a output directory under the following
    naming convention:

       <output_dir>/<dataset_name>/<source_language_code>_to_<target_language_code>/<checkpoint>/<template_name>/<date>/<split>.<file_type>

    Args:
        dataset (DatasetDict): A DatasetDict object of the original text dataset. Needs to have at least one split.
        dataset_name (str): Name of the dataset for storing output.
        template_name (str): Name of the template for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        target_language (str): the language you want translation to.
        url (str, optional): The URL of the inference API server. Defaults to "http://localhost:8000/translate".
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to cpu_count().
    """

    date = datetime.today().strftime('%Y-%m-%d')

    source_language_code = lang_name_to_code[source_language]
    target_language_code = lang_name_to_code[target_language]

    checkpoint_str = checkpoint.replace("/", "-")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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

        translation_path = os.path.join(output_dir, dataset_name, f"{source_language_code}_to_{target_language_code}",
                                        checkpoint_str, template_name, date)
        Path(translation_path).mkdir(exist_ok=True, parents=True)

        ds.to_csv(
            os.path.join(
                translation_path,
                f"{split}.csv",
            ),
            index=False,
        )
        ds.to_json(os.path.join(
            translation_path,
            f"{split}.jsonl",
        ))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
