from instructmultilingual.translate_datasets import translate_dataset_from_huggingface_hub

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

translate_dataset_from_huggingface_hub(
    repo_id = "CohereForAI/summarize_from_feedback",
    # train_set = ["en/xp3_piqa_None_train_finish_sentence_with_correct_choice.jsonl"],
    # validation_set = ["en/xp3_piqa_None_validation_finish_sentence_with_correct_choice.jsonl"],
    # test_set = [],
    dataset_name="summarize_from_feedback",
    template_name="temp",
    # splits=["train"],
    splits=["train", "validation"],
    translate_keys=["prompt", "chosen", "rejected"],
    # translate_keys=["prompt"],
    url= "http://localhost:8000/translate",
    output_dir= "/home/johndang_cohere_com/instruct-multilingual/full_translations",
    source_language= "English",
    translation_lang_codes=AYA_LANGUAGES,
    checkpoint="facebook/nllb-200-3.3B",
    num_proc= 8,
)