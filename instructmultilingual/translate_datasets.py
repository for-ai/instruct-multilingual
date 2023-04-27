"""Translate datasets from huggingface hub using a variety of methods"""

import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import requests
from google.cloud import translate_v2
from huggingface_hub import hf_hub_download

from datasets import DatasetDict, load_dataset
from instructmultilingual.cloud_translate_mapping import (cloud_translate_lang_code_to_name,
                                                          cloud_translate_lang_name_to_code)
from instructmultilingual.flores_200 import (lang_code_to_name, lang_name_to_code)

T5_LANG_CODES = [
    'afr_Latn', 'als_Latn', 'amh_Ethi', 'ace_Arab', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'ajp_Arab', 'apc_Arab',
    'arb_Arab', 'arb_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'bjn_Arab', 'kas_Arab', 'knc_Arab', 'min_Arab',
    'hye_Armn', 'azb_Arab', 'azj_Latn', 'eus_Latn', 'bel_Cyrl', 'ben_Beng', 'mni_Beng', 'bul_Cyrl', 'mya_Mymr',
    'cat_Latn', 'ceb_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'ces_Latn', 'dan_Latn', 'nld_Latn', 'eng_Latn',
    'epo_Latn', 'est_Latn', 'fin_Latn', 'fra_Latn', 'glg_Latn', 'kat_Geor', 'deu_Latn', 'ell_Grek', 'guj_Gujr',
    'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hun_Latn', 'isl_Latn', 'ibo_Latn', 'ind_Latn', 'gle_Latn',
    'ita_Latn', 'jpn_Jpan', 'jav_Latn', 'kan_Knda', 'kaz_Cyrl', 'khm_Khmr', 'kor_Hang', 'ckb_Arab', 'kmr_Latn',
    'kir_Cyrl', 'lao_Laoo', 'ace_Latn', 'bjn_Latn', 'knc_Latn', 'min_Latn', 'taq_Latn', 'lvs_Latn', 'lit_Latn',
    'ltz_Latn', 'mkd_Cyrl', 'plt_Latn', 'mal_Mlym', 'zsm_Latn', 'mal_Mlym', 'mlt_Latn', 'mri_Latn', 'mar_Deva',
    'khk_Cyrl', 'npi_Deva', 'nno_Latn', 'nob_Latn', 'pbt_Arab', 'pes_Arab', 'pol_Latn', 'por_Latn', 'ron_Latn',
    'rus_Cyrl', 'smo_Latn', 'gla_Latn', 'srp_Cyrl', 'sna_Latn', 'snd_Arab', 'sin_Sinh', 'slk_Latn', 'slv_Latn',
    'som_Latn', 'nso_Latn', 'sot_Latn', 'spa_Latn', 'sun_Latn', 'swh_Latn', 'swe_Latn', 'tgk_Cyrl', 'tam_Taml',
    'tel_Telu', 'tha_Thai', 'tur_Latn', 'ukr_Cyrl', 'urd_Arab', 'uzn_Latn', 'vie_Latn', 'cym_Latn', 'xho_Latn',
    'ydd_Hebr', 'yor_Latn', 'zul_Latn'
]

T5_CLOUD_TRANSLATE_LANG_CODES = [
    'jv', 'cy', 'ms', 'it', 'tg', 'sr', 'bs', 'zh-TW', 'km', 'la', 'sm', 'ee', 'am', 'be', 'kn', 'ht', 'gom', 'bm',
    'te', 'ig', 'ja', 'ts', 'ceb', 'et', 'sq', 'ga', 'lus', 'pl', 'hy', 'rw', 'lo', 'bn', 'da', 'dv', 'yi', 'lg', 'nso',
    'tt', 'is', 'ln', 'no', 'zu', 'eu', 'el', 'nl', 'so', 'haw', 'ko', 'ta', 'gd', 'eo', 'pa', 'ku', 'co', 'qu', 'fi',
    'kri', 'sl', 'sw', 'st', 'uk', 'lb', 'ur', 'ar', 'hi', 'ml', 'pt', 'cs', 'tk', 'sa', 'bho', 'mai', 'iw', 'ug', 'ak',
    'lt', 'bg', 'mi', 'sd', 'ny', 'ha', 'id', 'mg', 'fa', 'doi', 'ay', 'de', 'ky', 'om', 'es', 'ca', 'zh', 'vi', 'si',
    'gl', 'sv', 'ps', 'mk', 'yo', 'th', 'sk', 'af', 'or', 'mr', 'hr', 'su', 'mni-Mtei', 'hu', 'ka', 'ru', 'gu', 'gn',
    'lv', 'hmn', 'as', 'uz', 'en', 'ckb', 'ilo', 'ro', 'fr', 'mt', 'kk', 'jw', 'xh', 'fy', 'ti', 'zh-CN', 'sn', 'az',
    'tl', 'he', 'my', 'mn', 'tr', 'ne'
]


def inference_request(url: str, source_language: str, target_language:str, texts: List[str]) -> List[str]:
    """_summary_

    Args:
        url (str): _description_
        source_language (str): _description_
        target_language (str): _description_
        texts (List[str]): _description_

    Returns:
        List[str]: _description_
    """

    headers = {"Content-Type": "application/json"}
    data = {
        "source_language": source_language,
        "target_language": target_language,
        "texts": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["translated_texts"]


def call_inference_api(
    example: Dict[str,List[str]],
    url: str,
    source_lang_code: str,
    target_lang_code: str,
    keys_to_be_translated: List[str] =["dialogue", "summary"],
) -> Dict[str,List[str]]:
    """_summary_

    Args:
        example (Dict[str,List[str]]): _description_
        url (str): _description_
        source_lang_code (str): _description_
        target_lang_code (str): _description_
        keys_to_be_translated (List[str], optional): _description_. Defaults to ["dialogue", "summary"].

    Returns:
        Dict[str,List[str]]: _description_
    """
    for key in keys_to_be_translated:
        example[key] = inference_request(url, source_lang_code, target_lang_code, example[key])
    return example


def translate_dataset_via_inference_api(
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
    num_proc: int = 8,
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
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 8.
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
            lambda x: call_inference_api(
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


def cloud_translate(example: Dict[str, str],
                    target_lang_code: str,
                    keys_to_be_translated: List[str],
                    max_tries: int = 3) -> Dict[str, str]:
    """_summary_

    Args:
        example (Dict[str, str]): _description_
        target_lang_code (str): _description_
        keys_to_be_translated (List[str]): _description_
        max_tries (int, optional): _description_. Defaults to 3.

    Returns:
        Dict[str, str]: _description_
    """
    translate_client = translate_v2.Client()

    tries = 0

    while tries < max_tries:
        tries += 1
        try:
            for key in keys_to_be_translated:
                results = translate_client.translate(example[key], target_language=target_lang_code)
                example[key] = [result["translatedText"] for result in results]
                time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(2)

    return example


def translate_dataset_via_cloud_translate(
    dataset: DatasetDict,
    dataset_name: str,
    template_name: str,
    splits: List[str],
    translate_keys: List[str],
    target_language: str,
    output_dir: str = "./datasets",
    source_language: str = "English",
    checkpoint: str = "google_cloud_translate",
    num_proc: int = 8,
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
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 8.
    """

    date = datetime.today().strftime('%Y-%m-%d')

    source_language_code = cloud_translate_lang_name_to_code[source_language]
    target_language_code = cloud_translate_lang_name_to_code[target_language]

    checkpoint_str = checkpoint.replace("/", "-")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for split in splits:
        split_time = time.time()
        ds = dataset[split]
        print(f"[{split}] {len(ds)=}")
        ds = ds.map(
            lambda x: cloud_translate(
                x,
                target_lang_code=target_language_code,
                keys_to_be_translated=translate_keys,
            ),
            batched=True,
            batch_size=40,  # translate api has limit of 204800 bytes max at a time
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


def translate_dataset_from_huggingface_hub(dataset_name: str,
                                           template_name: str,
                                           splits: List[str],
                                           translate_keys: List[str],
                                           repo_id: str = "bigscience/xP3",
                                           train_set: List[str] = [],
                                           validation_set: List[str] = [],
                                           test_set: List[str] = [],
                                           url: str = "http://localhost:8000/translate",
                                           output_dir: str = "./datasets",
                                           source_language: str = "English",
                                           checkpoint: str = "facebook/nllb-200-3.3B",
                                           num_proc: int = 8,
                                           translation_lang_codes: List[str] = T5_LANG_CODES,
                                           exclude_languages: Set[str] = {"English"}) -> None:
    """A wrapper for using translate_dataset_via_api specifically on dataset
    repos from HuggingFace hub. The default repo is bigscience/xP3.

    Args:
        dataset_name (str): Name of the dataset for storing output.
        template_name (str): Name of the template for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        repo_id (str, optional): Name of the dataset repo on Huggingface. Defaults to "bigscience/xP3".
        train_set (List[str], optional): List of training set jsonl files for the dataset. Defaults to [].
        validation_set (List[str], optional): List of validation set jsonl files for the dataset. Defaults to [].
        test_set (List[str], optional): List of test set jsonl files for the dataset. Defaults to [].
        url (str, optional): The URL of the inference API server. Defaults to "http://localhost:8000/translate".
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 8.
        translation_lang_codes (List[str], optional): List of Flores-200 language codes to translate to. Defaults to T5_LANG_CODES.
        exclude_languages (Set[str], optional): Set of languages to exclude. Defaults to {"English"}.
    """
    assert len(train_set) > 0 or len(validation_set) > 0 or len(
        test_set) > 0, "Error: one of train/validation/test sets has to have a path"

    dataset_splits = {"train": train_set, "validation": validation_set, "test": test_set}

    dataset_template = defaultdict(list)

    temp_root = "temp_datasets"
    temp_dir = f"{temp_root}/{dataset_name}"
    Path(temp_dir).mkdir(exist_ok=True, parents=True)

    for split, files in dataset_splits.items():
        if len(files) > 0:
            temp_split_dir = f"{temp_root}/{dataset_name}/{split}"
            Path(temp_split_dir).mkdir(exist_ok=True, parents=True)
            for f in files:

                hf_hub_download(repo_id=repo_id,
                                local_dir=temp_split_dir,
                                filename=f,
                                repo_type="dataset",
                                local_dir_use_symlinks=False)

                pth = os.path.join(temp_split_dir, f)
                dataset_template[split].append(pth)
    print(dataset_template)
    dataset = load_dataset('json', data_files=dataset_template)

    # Make a copy of the source dataset inside translated datasets as well
    date = datetime.today().strftime('%Y-%m-%d')
    if checkpoint == "google_cloud_translate":
        source_language_code = cloud_translate_lang_name_to_code[source_language]
    else:
        source_language_code = lang_name_to_code[source_language]
    checkpoint_str = checkpoint.replace("/", "-")
    translation_path = os.path.join(output_dir, dataset_name, source_language_code, checkpoint_str, template_name, date)
    Path(translation_path).mkdir(exist_ok=True, parents=True)
    for s in dataset.keys():
        dataset[s].to_csv(
            os.path.join(
                translation_path,
                f"{s}.csv",
            ),
            index=False,
        )
        dataset[s].to_json(os.path.join(
            translation_path,
            f"{s}.jsonl",
        ))

    if checkpoint == "google_cloud_translate":
        for code in translation_lang_codes:
            l = cloud_translate_lang_code_to_name[code]
            if l not in exclude_languages:
                print(f"Currently translating: {l}")
                translate_dataset_via_cloud_translate(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    splits=splits,
                    translate_keys=translate_keys,
                    target_language=l,
                    output_dir=output_dir,
                    source_language=source_language,
                    checkpoint=checkpoint,
                    num_proc=num_proc,
                )
    else:
        for code in translation_lang_codes:
            l = lang_code_to_name[code]
            if l not in exclude_languages:
                print(f"Currently translating: {l}")
                translate_dataset_via_inference_api(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    splits=splits,
                    translate_keys=translate_keys,
                    target_language=l,
                    url=url,
                    output_dir=output_dir,
                    source_language=source_language,
                    checkpoint=checkpoint,
                    num_proc=num_proc,
                )
