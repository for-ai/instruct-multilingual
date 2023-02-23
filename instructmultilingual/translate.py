"""Translation."""

import fire
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from instructmultilingual.flores_200 import lang_name_to_code


def main(
    text: str = "Cohere For AI will make the best instruct multilingual model in the world",
    source_language: str = "English",
    target_language: str = "Egyptian Arabic",
) -> None:
    """
    text: (str) text need to be translated
    source_language: (str) the language of the text
    target_language: (str) the target language for the translation
    """
    source_language_code = lang_name_to_code[source_language]
    target_language_code = lang_name_to_code[target_language]

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-3.3B",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code],
        max_length=1024,
    )
    decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"Original Text ({source_language}): {text}")
    print(f"Translated Text ({target_language}): {decoded}")

    # >>> Original Text (English): Cohere For AI will make the best instruct multilingual model in the world
    # >>> Translated Text (Egyptian Arabic): Cohere For AI هيكون احسن نموذج تعليم متعدد اللغات في العالم


if __name__ == '__main__':
    fire.Fire(main)
