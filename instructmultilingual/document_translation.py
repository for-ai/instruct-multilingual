import os
import time
import argparse
import datasets
import ctranslate2 
from transformers import AutoTokenizer

class TranslationDataProcessor:
	"""
	Translation data processor class. 
	Objective: cache & tokenize data field/column
	"""
	def __init__(self, args):
		self.tokenizer = self.load_tokenizer(args.tokenizer_name_or_path, args.model_name_or_path)
		self.src_lang = args.src_lang
		self.tgt_lang = args.tgt_lang
		self.dataset = self.load_dataset(
	  			args.hf_dataset_path,
		 		args.hf_dataset_name,
		   		args.hf_dataset_data_dir,
			 	args.hf_dataset_data_files,
				args.hf_dataset_split
			)
	@classmethod
	def load_from_dataset_tokenizer(cls, dataset, tokenizer):
		cls.dataset = dataset
		cls.tokenizer = tokenizer
		cls.src_lang = tokenizer.src_lang
		cls.tgt_lang = tokenizer.tgt_lang
	
	def load_dataset(self, hf_dataset_path, hf_dataset_name, hf_dataset_data_dir, hf_dataset_data_files, hf_dataset_split):
		return datasets.load_dataset(
			hf_dataset_path,
			name = hf_dataset_name,
			data_dir = hf_dataset_data_dir,
			data_files = hf_dataset_data_files,
			split = hf_dataset_split
		)
  
	def load_tokenizer(self, tokenizer_name_or_path=None, model_name_or_path=None, src_lang=None, tgt_lang=None):
		if tokenizer_name_or_path is None:
			tokenizer_name_or_path = model_name_or_path
		tokenizer = AutoTokenizer.from_pretrained(
					tokenizer_name_or_path,
					src_lang=src_lang,
					tgt_lang=tgt_lang,
				)
		return tokenizer

	def _convert_to_features(self, example):
		tokenized = self.tokenizer.convert_ids_to_tokens(
	  		self.tokenizer.encode(
				example[self.tokenized_column_name]
			)
		)
		return {
			f"tokenized_{self.tokenized_column_name}" : tokenized
		}

	def preprocess(self, column_names, data=None, num_proc=1, batch_size = 1):
		if data is None:
			data = self.dataset
		for column_name in column_names:
			assert column_name in data.column_names
			self.tokenized_column_name = column_name
			data = data.map(self._convert_to_features, num_proc=num_proc, desc=f"Tokenizing column: {self.tokenized_column_name}")
		return data
	
class Translator():
	"""translate hf dataset.
	"""
	def __init__(self, args):
		self.model = self.load_model(args.model_type, args.model_name_or_path)
		self.tokenizer = self.load_tokenizer(args.tokenizer_name_or_path, args.model_name_or_path)
		self.dataset = None
		self.src_lang = None
		self.tgt_lang = None
  
	@classmethod
	def load_from_model_tokenizer_dataset(cls, model, tokenizer, dataset):
		cls.model = model
		cls.tokenizer = tokenizer
		cls.dataset = dataset
		cls.src_lang = tokenizer.src_lang
		cls.tgt_lang = tokenizer.tgt_lang

	@classmethod
	def load_from_model_tokenizer(cls, model, tokenizer):
		cls.model = model
		cls.tokenizer = tokenizer
		cls.src_lang = tokenizer.src_lang
		cls.tgt_lang = tokenizer.tgt_lang

	def load_model(self, model_type, model_name_or_path):
		if model_type == "ctranslate":
			translator = ctranslate2.Translator(
				model_name_or_path,
				device="cuda",
				compute_type="float16",
				device_index=[int(_id) for _id in os.environ['CUDA_VISIBLE_DEVICES'].split(",")],
			)
		else:
			raise NotImplementedError("Please implement model loader.")
		return translator

	def load_tokenizer(self, tokenizer_name_or_path=None, model_name_or_path=None, src_lang=None, tgt_lang=None):
		if tokenizer_name_or_path is None:
			tokenizer_name_or_path = model_name_or_path
		tokenizer = AutoTokenizer.from_pretrained(
					tokenizer_name_or_path,
					src_lang=src_lang,
					tgt_lang=tgt_lang,
				)
		return tokenizer

	def __translate_batch(self, batch_example):
		"batch translate"
		batched_tokenized_text = batch_example[self.translation_column_name]
		batched_tgt_lang = [self.tgt_lang] * len(batched_tokenized_text)
		translations = self.model.translate_batch(
			source = batched_tokenized_text,
			target_prefix = batched_tgt_lang,
			max_batch_size = len(batched_tokenized_text)
		)
		targets = [translation.hypotheses[0][1:] for translation in translations]
		batched_translation = {
	  		f"translation_{self.translation_column_name}": [self.tokenizer.convert_tokens_to_ids(target) for target in targets]
		}
		return batched_translation

	def translate(self, column_names, data, tgt_lang=None, batch_size=1):
		if tgt_lang is not None:
			self.tgt_lang = tgt_lang
		for column_name in column_names:
			trans_column_name = f"tokenized_{column_name}"
			assert trans_column_name in data.column_names
			self.translation_column_name = trans_column_name
			data = data.map(
       			self.__translate_batch, 
          		batched=True, 
          		batch_size = batch_size, 
            	num_proc = 1, 
             	desc = f"Translating column: {self.translation_column_name}"
            )
		return data


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--hf-dataset-path",
		type=str,
		default=None,
		help="""Path or name of the dataset. Depending on path, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.
				For local datasets:

				if path is a local directory (containing data files only) -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory e.g. './path/to/directory/with/my/csv/data'.
				if path is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory) -> load the dataset builder from the dataset script e.g. './dataset/squad' or './dataset/squad/squad.py'.
				For datasets on the Hugging Face Hub (list all available datasets and ids with datasets.list_datasets())

				if path is a dataset repository on the HF hub (containing data files only) -> load a generic dataset builder (csv, text etc.) based on the content of the repository e.g. 'username/dataset_name', a dataset repository on the HF hub containing your data files.
				if path is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory) -> load the dataset builder from the dataset script in the dataset repository e.g. glue, squad, 'username/dataset_name', a dataset repository on the HF hub containing a dataset script 'dataset_name.py'.
  		""",
	)
	parser.add_argument(
		"--hf-dataset-name",
		type=str,
		default=None,
		help="""the name of the dataset configuration.""",
	)
	parser.add_argument(
		"--hf-dataset-data_dir",
  		type=str,
		default=None,
		help="""Defining the data_dir of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and data_files is None, the behavior is equal to passing os.path.join(data_dir, **) as data_files to reference all the files in a directory.""",
	)
	parser.add_argument(
		"--hf-dataset-data_files",
		type=str,
		help=" Path(s) to source data file(s).",
	)
	parser.add_argument(
		"--hf-dataset-split",
		type=str,
		help="Which split of the data to load. If None, will return a dict with all splits (typically datasets.Split.TRAIN and datasets.Split.TEST). If given, will return a single Dataset. Splits can be combined and specified like in tensorflow-datasets.",
	)
	parser.add_argument(
		"--hf-dataset-trans-column",
		type=str,
		nargs="+",
		default=None,
		help="column/field name of huggingface dataset/jsonl/csv file that will be translated.",
	)
	parser.add_argument(
		"--src-lang",
		type=str,
		required=True,
		help="Source language.",
	)
	parser.add_argument(
		"--tgt-lang",
  		nargs="+",
		required=True,
		help="Target langugage.",
	)
	parser.add_argument(
		"--model-name-or-path",
  		type=str,
		default="models/nllb-200-3.3B-converted",
		help="Model name of path.",
	)
	parser.add_argument(
		"--model-type",
  		type=str,
		default="ctranslate",
		choices=["ctranslate"],
		help="Model type.",
	)
	parser.add_argument(
		"--tokenizer-name-or-path",
  		type=str,
		default=None, 
		help="hf-hokenizer name or path to the hf-tokenizer folder. "
  			 "If None, it will try loading by --model-name-or-path.",
	)
	parser.add_argument(
		"--batch-size",
  		type=int,
		default=1, 
		help="Batch size in translation api."
	)
	parser.add_argument(
		"--validation-process", 
  		default=None, 
		choices=["sequence_length",  "bert_score", "multilingual_sentence_similarity"],
	 	help="Validate the translation and add a similarity score."
	)
	parser.add_argument(
		"--back-translation", 
		action='store_true',
	 	help="It will backtranslate the translated sentence for validation and apply `--validation-process`."
	)
	parser.add_argument(
		"--num-proc", 
  		type=int, 
		default=1, 
	 	help="Total number of parallel process."
	)

	args = parser.parse_args()
	data_processor = TranslationDataProcessor(args)
	data = data_processor.preprocess(args.hf_dataset_trans_column, data=None, num_proc=args.num_proc)
	translator = Translator(args)
	data = translator.translate(args.hf_dataset_trans_column, data, args.tgt_lang, args.batch_size)


if __name__ == "__main__":
	main()