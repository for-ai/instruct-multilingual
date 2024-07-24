from instructmultilingual.translate_datasets import translate_dataset_from_huggingface_hub

translate_dataset_from_huggingface_hub(
    repo_id = "CohereForAI/mmlu_filtered_translation",
    train_set = [""],
    validation_set = [""],
    test_set = [],
    dataset_name="mmlu_filtered",
    template_name="gtranslate",
    splits=["train"],
    translate_keys=['question', 'option_0', 'option_1', 'option_2', 'option_3', 'answer'],
    url= "http://localhost:8000/translate",
    output_dir= "/home/shivalikasingh/instruct-multilingual/datasets",
    source_language= "English",
    checkpoint="google_cloud_translate", #"facebook/nllb-200-3.3B",
    num_proc= 32,
)