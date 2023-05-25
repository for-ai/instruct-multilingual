import os
import csv
import copy
import json
import tqdm
import argparse
import datasets
import subprocess
from datetime import date
import concurrent.futures
from typing import Tuple, Optional, List	
from promptsource.templates import Template
from .data_stat import SERIES_A_DATASET_NAME_DICT
datasets.logging.set_verbosity_error()

mt5_langs_name_pair = [
    ("Afrikaans", "af"), ("Albanian", "sq"), ("Amharic", "am"), ("Arabic", "ar"), ("Armenian", "hy"), ("Azerbaijani", "az"), 
    ("Basque", "eu"), ("Belarusian", "be"), ("Bengali", "bn"), ("Bulgarian","bg"), ("Burmese", "my"), 
    ("Catalan", "ca"), ("Cebuano", "ceb"), ("Chichewa", "ny"), ("Chinese", "zh"), ("Corsican", "co"), ("Czech", "cs"), 
    ("Danish", "da"), ("Dutch", "nl"), 
    ("English", "en"), ("Esperanto", "eo"), ("Estonian", "et"), 
    ("Filipino", "fil"), ("Finnish", "fi"), ("French", "fr"), 
    ("Galician", "gl"), ("Georgian", "ka"), ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"), 
    ("Haitian Creole", "ht"), ("Hausa", "ha"), ("Hawaiian", "haw"), ("Hebrew", "iw"), ("Hindi", "hi"), ("Hmong", "hmn"), ("Hungarian", "hu"), 
    ("Icelandic", "is"), ("Igbo", "ig"), ("Indonesian", "id"), ("Irish", "ga"), ("Italian", "it"), 
    ("Japanese", "ja"), ("Javanese", "jv"), 
    ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"), ("Korean", "ko"), ("Kurdish", "ku"), ("Kyrgyz", "ky"), 
    ("Lao", "lo"), ("Latin", "la"), ("Latvian", "lv"), ("Lithuanian", "lt"), ("Luxembourgish", "lb"), 
    ("Macedonian", "mk"), ("Malagasy", "mg"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"), ("Marathi", "mr"), ("Mongolian", "mn"), 
    ("Nepali", "ne"), ("Norwegian", "no"), 
    ("Pashto", "ps"), ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), 
    ("Romanian", "ro"), ("Russian", "ru"), 
    ("Samoan", "sm"), ("Scottish Gaelic", "gd"), ("Serbian", "sr"), ("Shona", "sn"), ("Sindhi", "sd"), ("Sinhala", "si"), ("Slovak","sk"), ("Slovenian", "sl"), ("Somali", "so"), ("Sotho", "st"), ("Spanish", "es"), ("Sundanese", "su"), ("Swahili", "sw"), ("Swedish", "sv"), 
    ("Tajik", "tg"), ("Tamil", "ta"), ("Telugu", "te"), ("Thai", "th"), ("Turkish", "tr"), 
    ("Ukrainian", "uk"), ("Urdu", "ur"), ("Uzbek", "uz"), 
    ("Vietnamese", "vi"), 
    ("Welsh", "cy"), ("West Frisian", "fy"), 
    ("Xhosa", "xh"), 
    ("Yiddish", "yi"), ("Yoruba", "yo"), ("Zulu", "zu")
]
mt5_langs_full_name_to_iso_name = { full_name: iso_name for full_name, iso_name in mt5_langs_name_pair}

dataset_mapper = {
	"AfriSenti-twitter-sentiment https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment": "shmuhammad/AfriSenti-twitter-sentiment",
	"Joke-explanation https://huggingface.co/datasets/theblackcat102/joke_explaination": "theblackcat102/joke_explaination",
	"Language Identification https://huggingface.co/datasets/papluca/language-identification": "papluca/language-identification",
	"Mafand - a machine translation task https://huggingface.co/datasets/masakhane/mafand": "sbmaruf/forai_ml_masakhane_mafand",
	"Masakhanews https://github.com/masakhane-io/masakhane-news": "masakhane/masakhanews",
	"Mintaka https://huggingface.co/datasets/AmazonScience/mintaka":"AmazonScience/mintaka",
	"NarrativeQA https://huggingface.co/datasets/narrativeqa": "narrativeqa",
	"NusaX - sentiment classification https://huggingface.co/datasets/indonlp/NusaX-senti": "indonlp/NusaX-senti",
	"qrecc https://huggingface.co/datasets/svakulenk0/qrecc": "svakulenk0/qrecc",
	"SODA https://huggingface.co/datasets/allenai/soda": "allenai/soda",
	"TED https://huggingface.co/datasets/ted_talks_iwslt": "sbmaruf/forai_ml-ted_talk_iwslt",
	"WikiCatSum https://huggingface.co/datasets/GEM/wiki_cat_sum": "GEM/wiki_cat_sum",
	"X-CSQA https://huggingface.co/datasets/xcsr": "xcsr",
	"xlel_wd https://huggingface.co/datasets/adithya7/xlel_wd": "adithya7/xlel_wd"
}

IGNORE_TASKS = [
	"arabic_billion_words",
	"narrativeqa",
	"svakulenk0/qrecc"
]
def check(
	json_example: str, 
	template_name: str, 
	jinja_template: str, 
	template_reference: Optional[str] = None, 
	original_task: Optional[str] = None, 
	choices_in_prompt: Optional[bool] = None,
	metrics: Optional[List[str]] = None,
	languages: Optional[List[str]] = None,
	answer_choices: Optional[str] = None
)-> Tuple[str, str]:
	"""
	Given an example (`json_example`) from a huggingface dataset and prompt template (`jinja_template`),
	the objective is to check if we can project the example in language model i/o format.  
	Args:
		json_example (str): a string contains json object. The json object is loaded 
								by `json.loads()`. Typically this is a sample from 
								huggingface dataset converted to a string by a `json.dumps()`. 
		template_name: unique name (per dataset) for template
        jinja_template: template expressed in Jinja
        template_reference: string describing author or paper reference for template
		original_task: If True, this prompt asks a model to perform the original task designed for
                this dataset.
		choices_in_prompt: If True, the answer choices are included in the templates such that models
			see those choices in the input. Only applicable to classification tasks.
		metrics: List of strings denoting metrics to use for evaluation
		languages: List of strings denoting languages used in the prompt (not the associated dataset!)
        answer_choices: Jinja expression for answer choices. Should produce
                            	a ||| delimited string of choices that enumerates
                            	the possible completions for templates that should
                            	be evaluated as ranked completions. If None, then
                            	the template is open-ended. This list is accessible
                            	from within Jinja as the variable `answer_choices`.
	"""
	json_example = json.loads(json_example)
	metadata = Template.Metadata(
		original_task,
		choices_in_prompt,
		metrics,
		languages
	)
	template = Template(
		template_name, 
	 	jinja_template, 
	  	template_reference, 
		metadata=metadata,
	   	answer_choices=answer_choices
	)
	lm_io = template.apply(json_example, highlight_variables=False)
	return lm_io
	

def create_name_with_hierarchy(output_dir, dataset_signature, dataset_subset, split_name, template_name, template_lang):
	"""
	<original-dataset-name>/<dataset_subset>__<split_name>/template-generation/<template>/<date>/foraiml__<original-dataset-name>__<dataset_subset>__<split>__<dataset_lang>__<prompt_lang>.jsonl
	"""
	split_lang = SERIES_A_DATASET_NAME_DICT[dataset_signature][dataset_subset]
	dataset_signature = dataset_signature.replace("/", "_").replace("\\","")
	dataset_uid = f"{dataset_signature}" 
	file_path = os.path.join(output_dir, dataset_uid)
	split_uid = f"{dataset_subset}__{split_name}"
	file_path = os.path.join(file_path, split_uid) 
	file_path = os.path.join(file_path, "template-generation") 
	file_path = os.path.join(file_path, template_name) 
	file_path = os.path.join(file_path, f"{date.today()}")
	dataset_file_uid = f"foraiml__{dataset_uid}__{split_uid}__{split_lang}__{mt5_langs_full_name_to_iso_name[template_lang]}.jsonl" 
	file_path = os.path.join(file_path, dataset_file_uid) 
	return file_path


def get_template_name(prompt_template_data):
	"""
		Prompt template named as the discord contributor.
		template_name: Name + discord_user_name
	"""
	name = prompt_template_data['Name'].replace(" ", "_").replace("#", "_") + \
			"__" + \
			prompt_template_data['Discord username'].replace(" ", "_").replace("#", "_")
	return name

def process(args):
	data, prompt_template_data, row_id, model_input, model_exp_output, export_file_path, generate, error_msg, generate_desc = args
	if generate:
		dir_name = os.path.dirname(export_file_path)
		if not os.path.exists(dir_name):
			os.makedirs(dir_name, exist_ok=True)
		if os.path.exists(export_file_path):
			print(f"[IGNORE] {export_file_path}: Data already exist.")
			return 0
		export_file_ptr = open(export_file_path, "w")
	
	for sample in tqdm.tqdm(data, total=len(data), desc=generate_desc):
		lm_io = check(
			json_example = json.dumps(sample),
			template_name = prompt_template_data['Name'],
			jinja_template = f"{model_input} ||| {model_exp_output}",
			template_reference = prompt_template_data['Discord username'],
		)
		assert len(lm_io) == 2, error_msg
		if not generate:
			break
		out_data = {
			"inputs": lm_io[0],
			"targets": lm_io[1],
		}
		export_file_ptr.write(f"{json.dumps(out_data)}\n")
	if generate:
		export_file_ptr.close()
	return 0

def validate_and_generate(prompt_template_data, row_id, output_dir="dumped", generate=False, num_proc=1):
	"""
	Generate data from a prompt template
	"""
	try:
		print(json.dumps(prompt_template_data, indent=4))
		dataset_info = prompt_template_data['What dataset do you pick?']
		if dataset_info not in dataset_mapper:
			dataset_signature = dataset_info.split()[0].lower()
		else:
			dataset_signature = dataset_mapper[dataset_info]
		if dataset_signature in IGNORE_TASKS:
			print(f"[IGNORE] row_id : {row_id}: Task exists in IGNORE_TASKS.")
			return 0
		if prompt_template_data['Automatic Generation'] != '1':
			print(f"[IGNORE] row_id : {row_id}: Automatic Generation = {prompt_template_data['Automatic Generation']}.")
			return 0
		dataset_subsets_dict = SERIES_A_DATASET_NAME_DICT[dataset_signature]
		template_name = get_template_name(prompt_template_data)
		with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as process_executor:
			future_to_val_results = []
			for dataset_subset_idx, (dataset_subset, subset_lang) in enumerate(dataset_subsets_dict.items()):
				dataset = datasets.load_dataset(dataset_signature, dataset_subset, verification_mode="no_checks")
				splits = dataset.keys()
				for split_idx, split in enumerate(splits):
					data = dataset[split]
					# validate & generate native lang prompt
					model_input = prompt_template_data['Input to the model']
					model_exp_output = prompt_template_data['Model\'s expected output']
					template_lang = prompt_template_data['What language do you want to write your prompt in?']
					export_file_path = create_name_with_hierarchy(output_dir, dataset_signature, dataset_subset, split, template_name, template_lang)
					error_msg = f"Validating dataset_signature:dataset_subset:split={dataset_signature}:{dataset_subset}:{split}:{row_id}:{template_lang} with prompt template... [FAILED]"
					generate_desc = f"row id: [{row_id}:{template_lang}] subset: [{dataset_subset_idx+1}/{len(dataset_subsets_dict)}]  split: [{split_idx+1}/{len(splits)}] {os.path.basename(export_file_path)}"
					# process(data, prompt_template_data, row_id, model_input, model_exp_output, export_file_path, generate, error_msg, generate_desc)
					
					args = (data, prompt_template_data, row_id, model_input, model_exp_output, export_file_path, generate, error_msg, generate_desc)
					future = process_executor.submit(process, args)
					future_to_val_results.append(future)

					# validate & generate english prompt
					model_input = prompt_template_data['English translation of the input']
					model_exp_output = prompt_template_data['English translation of the output']
					template_lang="English"
					export_file_path = create_name_with_hierarchy(output_dir, dataset_signature, dataset_subset, split, template_name, template_lang)
					error_msg = f"Validating dataset_signature:dataset_subset:split={dataset_signature}:{dataset_subset}:{split}:{row_id}:{template_lang} with prompt template... [FAILED]"
					generate_desc = f"row id: [{row_id}:{template_lang}] subset: [{dataset_subset_idx+1}/{len(dataset_subsets_dict)}]  split: [{split_idx+1}/{len(splits)}] {os.path.basename(export_file_path)}"
					# process(data, prompt_template_data, row_id, model_input, model_exp_output, export_file_path, generate, error_msg, generate_desc)
					
					args = (data, prompt_template_data, row_id, model_input, model_exp_output, export_file_path, generate, error_msg, generate_desc)
					future = process_executor.submit(process, args)
					future_to_val_results.append(future)

				print(f"[DONE] dataset_signature:dataset_subset:split={dataset_signature}:{dataset_subset}:{split}")
		concurrent.futures.wait(future_to_val_results)

	except:
		print(f"Error in row {row_id}, {dataset_signature=}, {dataset_subset=}")
		raise
	
	return 0
	
def parse(prompt_file_path, validate_rows):
	"""
	Parse list of rows menntioned in validate_rows. 
	"""
	_prmompt_dict, dt_structure, idx_to_header = {}, {}, {}
	with open(prompt_file_path, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row_idx, row in enumerate(csvreader):
			if row_idx == 0:
				for idx, dt in enumerate(row):
					dt_structure[dt] = {}
					idx_to_header[idx] = dt
			if row_idx+1 in validate_rows or validate_rows == []: # 1 based indexing
				sample = copy.deepcopy(dt_structure)
				for idx, dt in enumerate(row):
					sample[idx_to_header[idx]] = dt
				_prmompt_dict[ row_idx+1 ] = sample
	return _prmompt_dict


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--form_path",
		type=str,
		default="https://docs.google.com/spreadsheets/d/10bCwOhM8zKNkqKi54gIvdwrR44YlWQFV9fpGm7acHv8/export?format=csv&id=10bCwOhM8zKNkqKi54gIvdwrR44YlWQFV9fpGm7acHv8&gid=726399306",
		help="Path of the google sheet."
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite eexisting prompt file prompts.csv."
	)
	parser.add_argument(
		"--prompt-dir",
		type=str,
		default="data/",
		help="Overwrite existing prompt file prompts.csv."
	)
	parser.add_argument(
		"--validate-rows",
		nargs='*',
		default=[],
		type=int,
		help="List of row indices (1-based indexing ). The row mentioned here will indicate the row of `--form_path` spreadsheet. If empty, it will validate all the rows."
	)
	parser.add_argument(
		"--generate",
		action="store_true",
		help="Generate projected samples."
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="The path to the folder where data will be saved."
	)
	parser.add_argument(
		"--num-proc",
		default=1,
		type=int,
		help="Number of parallel process to run."
	)
	args = parser.parse_args()
	
	prompt_file_path = f"{args.prompt_dir}/prompts.csv"
	if not os.path.exists(prompt_file_path):
		os.makedirs(prompt_file_path)

	if os.path.exists(prompt_file_path) and args.overwrite: # if file exists, it may be from prev. run/download.
		subprocess.check_output(f"mv {prompt_file_path} {prompt_file_path}.old", shell=True)
	if not os.path.exists(prompt_file_path):
		cmd = f"curl -L '{args.form_path}' -o {prompt_file_path}"
		subprocess.check_output(cmd, shell=True)

	prompt_dict = parse(prompt_file_path, args.validate_rows)
	for row_id, prompt_template_data in prompt_dict.items():
		print(f"Working on row {row_id} ...")
		validate_and_generate(
			prompt_template_data, 
			row_id, 
			output_dir=args.output_dir, 
			generate=args.generate, 
			num_proc=args.num_proc
		)



if __name__ == "__main__":
	main()