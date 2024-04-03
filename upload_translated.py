import pandas as pd
import datasets

path = ("/home/johndang_cohere_com/instruct-multilingual/datasets_new_sub/summarize_from_feedback/eng_Latn_to_{lang}/facebook-nllb-200-3.3B/temp/2024-03-26/{split}.csv")

AYA_LANGUAGES = [
    "arb_Arab", 
    "zho_Hans", 
    "ces_Latn", 
    "nld_Latn",
    "fra_Latn", 
    "deu_Latn", 
    "ell_Grek", 
    "hin_Deva", 
    "ind_Latn", 
    "ita_Latn", 
    "jpn_Jpan", 
    "kor_Hang", 
    "pes_Arab", 
    "pol_Latn", 
    "por_Latn", 
    "ron_Latn", 
    "rus_Cyrl", 
    "spa_Latn", 
    "tur_Latn", 
    "ukr_Cyrl", 
    "vie_Latn"
]

AYA_LANGUAGES_TO_CODE = {
    "arb_Arab" : "ar", 
    "zho_Hans": "zh", 
    "ces_Latn" : "cs", 
    "nld_Latn" : "nl",
    "fra_Latn" : "fr", 
    "deu_Latn" : "de", 
    "ell_Grek" : "el", # #
    "hin_Deva" : "hi", 
    "ind_Latn" : "id", 
    "ita_Latn" : "it", 
    "jpn_Jpan" : "ja", 
    "kor_Hang" : "ko", 
    "pes_Arab" : "fa", 
    "pol_Latn" : "pl",  # # 
    "por_Latn" : "pt", 
    "ron_Latn" : "ro", 
    "rus_Cyrl" : "ru", 
    "spa_Latn" : "es", 
    "tur_Latn" : "tr", 
    "ukr_Cyrl" : "uk", 
    "vie_Latn" : "vi",
    "eng_Latn" : "en"
}

# langs = AYA_LANGUAGES
langs = ["eng_Latn"]
splits = [
    "train", "validation"
]
for l in langs:
    lang_ds = {}
    for s in splits:
        df = pd.read_csv(path.format(lang=l, split=s))
        # print(df.head())

        ds = datasets.Dataset.from_pandas(df)
        lang_ds[s] = ds

    lang_ds = datasets.DatasetDict(lang_ds)
    lang_ds.push_to_hub("CohereForAI/summarize_from_feedback_translated_sub", AYA_LANGUAGES_TO_CODE[l], private=True)
