export CUDA_VISIBLE_DEVICES='0,1,2'
python translate_tweak.py \
--hf-dataset-path "cnn_dailymail" \
--hf-dataset-name "3.0.0" \
--hf-dataset-split "train" \
--src-lang "eng_Latn" \
--tgt-lang "arz_Arab" \
--model-name-or-path "models/nllb-200-3.3B-converted" \
--model-type "ctranslate" \
--tokenizer-name-or-path "facebook/nllb-200-3.3B" \
--hf-dataset-trans-column "article" "highlights" \
--batch-size 64 \
--num-proc 50
