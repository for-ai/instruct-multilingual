import os
import csv
import copy
import json
import argparse
import datasets
import subprocess
from typing import Tuple, Optional, List	
from promptsource.templates import Template, LANGUAGES
from .data_stat import SERIES_A_DATASET_NAME_DICT

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


def validate(prompt_template_data, row_id):
	"""
	Validate a prompt template
	"""
	try:
		print(json.dumps(prompt_template_data, indent=4))
		dataset_info = prompt_template_data['What dataset do you pick?']
		if dataset_info not in dataset_mapper:
			dataset_signature = dataset_info.split()[0].lower()
		else:
			dataset_signature = dataset_mapper[dataset_info]
		dataset_subsets = SERIES_A_DATASET_NAME_DICT[dataset_signature]
		for dataset_subset in dataset_subsets:
			dataset = datasets.load_dataset(dataset_signature, dataset_subset)
			splits = dataset.keys()
			for split in splits:
				data = dataset[split]
				model_input = prompt_template_data['Input to the model']
				model_exp_output = prompt_template_data['Model\'s expected output']
				for sample in data:
					lm_io = check(
						json_example = json.dumps(sample),
						template_name = prompt_template_data['Name'],
						jinja_template = f"{model_input} ||| {model_exp_output}",
						template_reference = prompt_template_data['Discord username'],
					)
					if len(lm_io) == 2:
						print(f"Validating dataset_signature:dataset_subset:split={dataset_signature}:{dataset_subset}:{split} with prompt template... [DONE]")
					else:
						print(f"Validating dataset_signature:dataset_subset:split={dataset_signature}:{dataset_subset}:{split} with prompt template... [FAILED]")
						raise ValueError("Templating Error.")
					break
	except:
		print(f"Error in row {row_id}")
		raise
	
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
			if row_idx+1 in validate_rows: # 1 based indexing
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
		default=[3],
		type=int,
		help="List of row indices (1-based indexing ). The row mentioned here will indicate the row of `--form_path` spreadsheet."
	)
	args = parser.parse_args()
	prompt_file_path = f"{args.prompt_dir}/prompts.csv"
	if os.path.exists(prompt_file_path) and args.overwrite: # if file exists, it may be from prev. run/download.
		subprocess.check_output(f"mv {prompt_file_path} {prompt_file_path}.old", shell=True)
	if not os.path.exists(prompt_file_path):
		cmd = f"curl -L '{args.form_path}' -o {prompt_file_path}"
		subprocess.check_output(cmd, shell=True)

	prompt_dict = parse(prompt_file_path, args.validate_rows)
	for row_id, prompt_template_data in prompt_dict.items():
		print(f"Validating row {row_id} ...")
		validate(prompt_template_data, row_id)



if __name__ == "__main__":
	main()