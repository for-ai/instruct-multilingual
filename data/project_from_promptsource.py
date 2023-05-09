import os
import json
import logging
import argparse
import datasets
from tqdm import tqdm
import concurrent.futures
from typing import Type, Union, List, Optional
from tqdm.contrib.concurrent import process_map
from promptsource.templates import DatasetTemplates, Template
from datasets import Dataset, DatasetDict, IterableDatasetDict, IterableDataset

logger = logging.getLogger(__name__)


def export_dataset(
	dataset_output_dir: str,
	dataset_name: str,
	dataset_config: str,
	psrc_prompt_template_signature: str,
	prompt_template: Type[Template],
	dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
	add_source_metadata: bool = False,
	highlight_variables: bool = False,
	lang: str = 'en'
) -> str:
	"""
	Given a `hf-dataset` (arg: dataset) and a prompt template (arg: prompt_template),
	project/transform samples from all the splits of dataset (arg: dataset) into an instruction format and
	writes in the disk (arg: dataset_output_dir)

	Args:
		dataset_output_dir (str): Path to the output directory where data will be saved.
		dataset_name (str): Name of the hf-dataset.
		dataset_config (str): Name of the hf-dataset config.
		psrc_prompt_template_signature (str): Name of the dataset & dataset-config for which prompts are written for.
		prompt_template (Type[Template]): Transformation/projection module that will take a sample from arg:dataset and transform it to an instruction.
		dataset (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): huggingface dataset that will be transformed into an instruction dataset.
		add_source_metadata (bool = False): If True, all the data column from the args:dataset will be saved as a meta information with the instruction dataset.
		highlight_variables (bool = False): If True, prompt tokens and dataset tokens will be highlighted differently. This metadata will be saved as  `highlighted_source` & `highlighted_target`.
		lang (str = 'en'): language name of the dataset
	"""
	splits = list(dataset.keys())
	prompt_name = prompt_template.get_name()
	for split in splits:
		dataset_split = dataset[split]
		json_data_path = os.path.join(dataset_output_dir, split)
		os.makedirs(json_data_path, exist_ok=True)
		json_data_path = os.path.join(
			json_data_path,
			(psrc_prompt_template_signature + "." + prompt_name)
			.replace("/", "_")
			.replace(" ", "_")
			+ f"_{lang}.jsonl",
		)
		with open(json_data_path, "w", encoding="utf-8") as file_ptr:
			total_num_sample = len(dataset_split)
			for _id, sample in tqdm(
				enumerate(dataset_split),
				total=total_num_sample,
				desc="{}_{}_{}_{}_{}".format(
					dataset_name,
					dataset_config,
					split,
					psrc_prompt_template_signature,
					prompt_name,
				),
			):
				# Project/transform sample into instruction.
				prompted_sample = prompt_template.apply(
					sample, highlight_variables=False
				)
				answer_choice_list = prompt_template.get_answer_choices_list(
					sample
				)  # set of potential outcomes.
				if (
					len(prompted_sample) != 2
				):  # if the prompt doesn't generate a tuple, that means it's an invalid prompted_sample
					continue
				source, target = prompted_sample
				projected_sample_with_metadata = {
					"id": _id,  # An unique id for the sample. Each line of the `jsonl` file contains `json` data which has a unique id within the `jsonl` file. (datatype: string/int)
					"source": source,  # projected input for the language model. This is the instruction. (datatype: string)
					"target": target,  # projected output for the language model. This is the gold response. (datatype: string)
					"psrc_prompt_template_signature": psrc_prompt_template_signature,  # prompt template signature from promptsource repository. Usually, a set of prompt templates are written for a task (i.e., glue/cola, glue/mrpc). This usually refers to that task. (datatype: string)
					"prompt_name": prompt_name,  #  Name of the individual prompt template.  Under a `psrc_prompt_template_signature` there could be many prompt templates. `prompt_name` refers to each of those prompt templates. (datatype: string)
					"prompt_answer_choice_list": answer_choice_list,  # Name of all potential outcomes. We often do not have any data for this field. Especially for generative tasks. Only categorical task has this field (i.e., [yes, no], [True, False], [A, B, C, D]). (datatype: list of strings)
					"dataset_name": dataset_name,  # Name of the huggingface dataset  (datatype: string)
					"dataset_config": dataset_config,  # Subset name of the huggingface dataset (datatype: string)
					"split": split,  # Split name (i.e., train, dev, test) (datatype: string)
					"metrics": prompt_template.metadata.metrics,  # metrics to evaluate the response. (datatype: list of strings)
					"original_task": prompt_template.metadata.original_task,  # If the prompted sample (source, target) refers to the original task for the dataset being created (datatype: True/False)
					"choices_in_prompt": prompt_template.metadata.choices_in_prompt,  # If there is any randomness in the prompt generation (datatype: list of strings)
					"languages": prompt_template.metadata.languages,  # The language of the prompt template (not the dataset). (datatype: list of strings)
				}
				if highlight_variables:
					# Add highlight between prompt tokens and dataset tokens.
					new_projected_sample = prompt_template.apply(
						sample, highlight_variables=highlight_variables
					)
					source, target = new_projected_sample
					projected_sample_with_metadata["highlighted_source"] = source
					projected_sample_with_metadata["highlighted_target"] = target

				if add_source_metadata:
					#  Take a backup of the data columns of the original dataset.
					#  This will help us to recover original projection in case we loose track of the generated ones due to various modifications & filters.
					for k, v in sample.items():
						k = "src_meta_{}".format(k)
						assert k not in projected_sample_with_metadata
						projected_sample_with_metadata[k] = v

				file_ptr.write(json.dumps(projected_sample_with_metadata))
				file_ptr.write("\n")
	return "Completed:: {} !".format(json_data_path)


def xp3_export_dataset(
	dataset_output_dir: str,
	dataset_name: str,
	dataset_config: str,
	psrc_prompt_template_signature: str,
	prompt_template: Type[Template],
	dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
	lang: str = 'en'
) -> str:
	"""
	Given a `hf-dataset` (arg: dataset) and a prompt template (arg: prompt_template),
	project/transform samples from all the splits of dataset (arg: dataset) into an instruction format and
	writes in the disk (arg: dataset_output_dir)

	Args:
		dataset_output_dir (str): Path to the output directory where data will be saved.
		dataset_name (str): Name of the hf-dataset.
		dataset_config (str): Name of the hf-dataset config.
		psrc_prompt_template_signature (str): Name of the dataset & dataset-config for which prompts are written for.
		prompt_template (Type[Template]): Transformation/projection module that will take a sample from arg:dataset and transform it to an instruction.
		dataset (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): huggingface dataset that will be transformed into an instruction dataset.
		lang (str = 'en'): language name of the dataset
	"""
	splits = list(dataset.keys())
	prompt_name = prompt_template.get_name()
	for split in splits:
		dataset_split = dataset[split]
		json_data_path = os.path.join(dataset_output_dir, split)
		os.makedirs(json_data_path, exist_ok=True)
		json_data_path = os.path.join(
			json_data_path,
			f"foraiml_{dataset_name}_{lang}_{prompt_name}.jsonl"
		)
		with open(json_data_path, "w", encoding="utf-8") as file_ptr:
			total_num_sample = len(dataset_split)
			for _id, sample in tqdm(
				enumerate(dataset_split),
				total=total_num_sample,
				desc="{}_{}_{}_{}_{}".format(
					dataset_name,
					dataset_config,
					split,
					psrc_prompt_template_signature,
					prompt_name,
				),
			):
				# Project/transform sample into instruction.
				prompted_sample = prompt_template.apply(
					sample, highlight_variables=False
				)
				answer_choice_list = prompt_template.get_answer_choices_list(
					sample
				)  # set of potential outcomes.
				if (
					len(prompted_sample) != 2
				):  # if the prompt doesn't generate a tuple, that means it's an invalid prompted_sample
					continue
				source, target = prompted_sample
				projected_sample_with_metadata = {
					"inputs": source,  # projected input for the language model. This is the instruction. (datatype: string)
					"targets": target,  # projected output for the language model. This is the gold response. (datatype: string)
				}

				file_ptr.write(json.dumps(projected_sample_with_metadata))
				file_ptr.write("\n")
	return "Completed:: {} !".format(json_data_path)


def invoke_none(lst: List[str]) -> Union[List[str], None]:
	"""
	helper function.
	Takes a list of string and replace `None` where needed.
	"""
	for idx, val in enumerate(lst):
		if val == "None" or val == "none" or val == "null" or val == "":
			lst[idx] = None
	return lst


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset-name-or-paths",
		nargs="+",
		default="glue",
		help="""A list of paths (seperated by space) to a huggingface dataset (or huggingface dataset singnature, i.e, super_glue, squad_v2).
		A supported list can be found at https://github.com/bigscience-workshop/promptsource/tree/main/promptsource/templates .
		Usually prompt templates are written for a specific datasets. But in the case of a new dataset, 
		it is possible to apply a different (written for a different dataset) prompt template to a new dataset as long as 
		the JSON structure of the dataset is the same as what is required in the original prompt template.""",
	)
	parser.add_argument(
		"--dataset-configs",
		nargs="+",
		default=None,
		help="""A list of huggingface dataset-config. `--dataset-name-or-paths` along with `--dataset-configs` defines a data file.
		If there is no `--dataset-configs` in huggingface, use None. The first argument in the `--dataset-name-or-paths` refers to the 
		first argument of the `--dataset-configs`. There should be an equal number of argument in `--dataset-name-or-paths` and `--dataset-configs`.""",
	)
	parser.add_argument(
		"--prompt-templates-configs",
		nargs="+",
		default=None,
		help="""Name of the prompt template. Please use `None` if you want to project with all the prompt templates. 
		The first argument in the `--dataset-name-or-paths` & `--dataset-configs` refers to the 
		first argument of the `--prompt-templates-configs`. There should be an equal number of argument in 
		`--dataset-name-or-paths`, `--dataset-configs` and `--prompt-templates-configs`""",
	)
	parser.add_argument(
		"--cache-dir",
		type=str,
		required=True,
		help="Path to the cache dir of huggingface datasets. (The directory may require very large space.)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		required=True,
		help="Path to the output dir where the projected data will be stored.",
	)
	parser.add_argument(
		"--num-proc", type=int, default=9, help="Total number of parallel process."
	)
	parser.add_argument(
		"--add-source-metadata",
		action="store_true",
		help="""
		Add all the metadata from source dataset. This will create new keys names `src_meta_{original_keys}` 
		where this `original_keys` are all the keys from the original dataset key names (a.k.a column name).
		These variable are kept with the completion so that we can recover the projection again if needed.
		""",
	)
	parser.add_argument(
		"--highlight-variables",
		action="store_true",
		help="""Highlight token that are coming from the prompts and original dataset."
		This feature can be used to differentiate prompt tokens and input tokens.""",
	)
	parser.add_argument(
		"--xp3-format",
		action="store_true",
		help="""Export the data in xP3 format""",
	)
	parser.add_argument(
		"--lang",
		type=str,
		default='en',
		help="""Language name. Required for xP3 naming of the file.""",
	)
	args = parser.parse_args()

	assert len(args.dataset_name_or_paths) == len(args.dataset_configs)
	assert len(args.dataset_name_or_paths) == len(args.prompt_templates_configs)
	export_dataset_func = xp3_export_dataset if args.xp3_format else export_dataset
	if args.xp3_format and args.highlight_variables:
		print(f"Ignoring {args.highlight_variables=} since {args.xp3_format}")
	if args.xp3_format and args.add_source_metadata:
		print(f"Ignoring {args.add_source_metadata=} since {args.xp3_format}")
	
	invoke_none(args.dataset_name_or_paths)
	invoke_none(args.dataset_configs)
	invoke_none(args.prompt_templates_configs)

	prompted_sample_gen_io_tuple_list = []
	# loading and caching each of the dataset & creating multiprocessor i/o for doing projection.
	for (dataset_name_or_path, dataset_config, prompt_template_config) in zip(
		args.dataset_name_or_paths, args.dataset_configs, args.prompt_templates_configs
	):
		dataset = datasets.load_dataset(
			dataset_name_or_path, dataset_config, cache_dir=args.cache_dir
		)
		psrc_prompt_template_signature = prompt_template_config
		if psrc_prompt_template_signature is None:
			if dataset_config is None:
				psrc_prompt_template_signature = "{}".format(dataset_name_or_path)
			else:
				psrc_prompt_template_signature = "{}/{}".format(
					dataset_name_or_path, dataset_config
				)
		dataset_output_dir = os.path.join(args.output_dir, dataset_name_or_path)
		os.makedirs(dataset_output_dir, exist_ok=True)
		if dataset_config is not None:
			dataset_output_dir = os.path.join(dataset_output_dir, dataset_config)
			os.makedirs(dataset_output_dir, exist_ok=True)
		prompt_templates = DatasetTemplates(psrc_prompt_template_signature)
		prompt_names = list(prompt_templates.name_to_id_mapping.keys())
		for prompt_name in prompt_names:
			prompt_template = prompt_templates[prompt_name]
			# pre-calculate the arguments for multiprocesssing.
			prompted_sample_gen_io_tuple = (
				dataset_output_dir,
				dataset_name_or_path,
				dataset_config,
				psrc_prompt_template_signature,
				prompt_template,
				dataset,
				args.add_source_metadata,
				args.highlight_variables,
			)
			prompted_sample_gen_io_tuple_list.append(prompted_sample_gen_io_tuple)

	# Projecting data using multiprocessing.
	# It's recommended to use large number of CPU machine if you are projecting multiple dataset.
	# set up `--num-proc` accrodingly.
	num_proc = min(args.num_proc, len(prompted_sample_gen_io_tuple_list))
	with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
		for _out in tqdm(
			executor.map(export_dataset_func, *zip(*prompted_sample_gen_io_tuple_list)),
			total=len(args.dataset_name_or_paths),
		):
			try:
				logger.info(_out)
			except Exception as emsg:
				logger.warning("Exception msg: {}".format(emsg))


if __name__ == "__main__":
	main()
