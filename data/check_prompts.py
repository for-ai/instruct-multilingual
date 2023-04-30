import os
import csv
import json
import argparse
import subprocess
from typing import Tuple, Optional	
from promptsource.templates import Template
from .data_stat import SERIES_A_DATASET_NAME_DICT

def check(
	json_example: str, 
	template_name: str, 
	jinja_template: str, 
	template_reference: Optional[str] = None, 
	answer_choices: Optional[str] = None
)-> Tuple[str, str]:
	"""
	Given a 
	Args:
		json_example (str): a string contains json object. The json object is loaded 
								by `json.loads()`. Typically this is a sample from 
								huggingface dataset converted to a string by a `json.dumps()`. 
		template_name: unique name (per dataset) for template
        jinja_template: template expressed in Jinja
        template_reference: string describing author or paper reference for template
        answer_choices: Jinja expression for answer choices. Should produce
                            	a ||| delimited string of choices that enumerates
                            	the possible completions for templates that should
                            	be evaluated as ranked completions. If None, then
                            	the template is open-ended. This list is accessible
                            	from within Jinja as the variable `answer_choices`.
	"""
	json_example = json.loads(json_example)
	template = Template(
		template_name, 
	 	jinja_template, 
	  	template_reference, 
	   	answer_choices=answer_choices
	)
	lm_io = template.apply(json_example, highlight_variables=False)
	return lm_io

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--form_path",
		type=str,
		default=None,
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
		help="Overwrite eexisting prompt file prompts.csv."
	)
	args = parser.parse_args()
	prompt_file_path = f"{args.prompt_dir}/prompts.csv"
	if os.path.exists(prompt_file_path) and args.overwrite: # if file exists, it may be from prev. run/download.
		subprocess.check_output(f"mv {prompt_file_path} {prompt_file_path}.old", shell=True)
		subprocess.check_output("curl -L https://docs.google.com/spreadsheets/d/10bCwOhM8zKNkqKi54gIvdwrR44YlWQFV9fpGm7acHv8/export?format=csv > ./data/prompts.csv", shell=True)
  
	with open('data/prompts.csv', 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		next(iter(csvreader))
		for row in csvreader:
			print(row)
 
if __name__ == "__main__":
	main()