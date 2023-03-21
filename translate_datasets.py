import os
import pandas as pd

from datasets import load_dataset
from instructmultilingual.flores_200 import lang_name_to_code
from optimum.onnxruntime import ORTModel, ORTModelForSeq2SeqLM
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List


    
def translate_dataset(
        dataset_name: str,
        splits: List[str],
        translate_cols:List[str],
        output_dir: str,
        target_languages: List[str],
        source_language: str = "English",
        checkpoint: str = "facebook/nllb-200-3.3B",
        device: str = "cuda:0",
        printout: bool = False,
    ) -> None:

    datasets = []
    for split in splits:
        datasets.append(load_dataset(dataset_name, split=split))

    if "onnx" in checkpoint:
        model = ORTModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    model.to(device)

    for dataset in datasets:
        for target_language in target_languages:
            translated_dataset = dataset.to_pandas()

            source_language_code = lang_name_to_code[source_language]
            target_language_code = lang_name_to_code[target_language]

            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                src_lang=source_language_code,
                tgt_lang=target_language_code,
            )
            

            for i in range(len(dataset)):
                for col in translate_cols:
                    text = translated_dataset.loc[i, col]
                    
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    translated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code],
                        max_length=1024,
                    )
                    decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    translated_dataset.loc[i, f"{col}_{target_language}"] = decoded
                    if printout:
                        print(f"{i}| Original ({source_language}): {text}")
                        print(f"{i}| Translated ({target_language}): {decoded}")

            checkpoint_str = checkpoint.replace("/", "-") 
            translated_dir = Path(os.path.join(output_dir, dataset_name, target_language_code))
            translated_dir.mkdir(parents=True, exist_ok=True)
            translated_dataset.to_csv(os.path.join(translated_dir, f"{dataset_name}_{split}_{target_language_code}_{checkpoint_str}.csv"), index=False)
            
    


if __name__ == "__main__":
    root = "/home/weiyi/instruct-multilingual/datasets"

    # translate_dataset(
    #     "rotten_tomatoes", 
    #     ["validation"],
    #     ["text"],
    #     root, 
    #     [
    #         # "Egyptian Arabic",
    #         # "Bengali",
    #         # "Western Persian",
    #         "Chinese (Traditional)",
    #         "Portuguese",
    #     ],
    #     # checkpoint="facebook/nllb-200-distilled-600M",
    #     checkpoint="facebook/nllb-200-3.3B",
    #     # printout=True
    # )

    translate_dataset(
        "samsum", 
        ["validation"],
        ["dialogue", "summary"],
        root, 
        [
            # "Egyptian Arabic",
            # "Bengali",
            # "Western Persian",
            # "Chinese (Traditional)",
            "Chinese (Simplified)",
            # "Portuguese",
        ],
        # checkpoint="facebook/nllb-200-distilled-600M",
        checkpoint="facebook/nllb-200-3.3B",
        # checkpoint="onnx_facebook_nllb-200-3.3B/",
        printout=True
    )
