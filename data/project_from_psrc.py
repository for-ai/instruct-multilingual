import os
import json
import argparse
import datasets
from tqdm import tqdm
import concurrent.futures
from tqdm.contrib.concurrent import process_map
from promptsource.templates import DatasetTemplates



def export_dataset(
	dataset_output_dir,
	dataset_name,
	dataset_config,
	psrc_prompt_template_signature,
	prompt,
	dataset,
	add_source_metadata=False,
	highlight_variables=False,
):
	splits = list(dataset.keys())
	prompt_name = prompt.get_name()
	for split in splits:
		dataset_split = dataset[split]
		json_data_path = os.path.join(dataset_output_dir, split)
		os.makedirs(json_data_path, exist_ok=True)
		json_data_path = os.path.join(
			json_data_path,
			(psrc_prompt_template_signature + "." + prompt_name).replace("/", "_").replace(" ", "_")
			+ ".jsonl",
		)
		with open(json_data_path, "w", encoding="utf-8") as file_ptr:
			total_num_sample = len(dataset_split)
			for _id, sample in tqdm(
				enumerate(dataset_split),
				total=total_num_sample,
				desc="{}_{}_{}_{}_{}".format(
					dataset_name, dataset_config, split, psrc_prompt_template_signature, prompt_name
				),
			):
				projected_sample = prompt.apply(sample, highlight_variables=False)
				answer_choice_list = prompt.get_answer_choices_list(sample)
				if len(projected_sample) != 2:
					continue
				source, target = projected_sample
				projected_sample_with_metadata = {
					"id": _id,
					"source": source,
					"target": target,
					"psrc_prompt_template_signature": psrc_prompt_template_signature,
					"prompt_name": prompt_name,
					"prompt_answer_choice_list": answer_choice_list,
					"dataset_name": dataset_name,
					"dataset_config": dataset_config,
					"split": split,
					"metrics": prompt.metadata.metrics,
					"original_task": prompt.metadata.original_task,
					"choices_in_prompt": prompt.metadata.choices_in_prompt,
					"languages": prompt.metadata.languages,
				}
				if highlight_variables:
					new_projected_sample = prompt.apply(
						sample, highlight_variables=highlight_variables
					)
					source, target = new_projected_sample
					projected_sample_with_metadata["highlighted_source"] = source
					projected_sample_with_metadata["highlighted_target"] = target

				if add_source_metadata:
					for k, v in sample.items():
						k = "src_meta_{}".format(k)
						assert k not in projected_sample_with_metadata
						projected_sample_with_metadata[k] = v

				file_ptr.write(json.dumps(projected_sample_with_metadata))
				file_ptr.write("\n")
	return "Completed:: {} !".format(json_data_path)


def invoke_none(lst):
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
		the JSON structure of the dataset is the same as what is required in the original prompt template."""
	)
	parser.add_argument(
		"--dataset-configs",
		nargs="+",
		default=None,
		help="""A list of huggingface dataset-config. `--dataset-name-or-paths` along with `--dataset-configs` defines a data file.
		If there is no `--dataset-configs` in huggingface, use None. The first argument in the `--dataset-name-or-paths` refers to the 
		first argument of the `--dataset-configs`. There should be an equal number of argument in `--dataset-name-or-paths` and `--dataset-configs`."""
	)
	parser.add_argument(
		"--prompt-templates-configs",
		nargs="+",
		default=None,
		help="""Name of the prompt template. Please use `None` if you want to project with all the prompt templates. 
		The first argument in the `--dataset-name-or-paths` & `--dataset-configs` refers to the 
		first argument of the `--prompt-templates-configs`. There should be an equal number of argument in 
		`--dataset-name-or-paths`, `--dataset-configs` and `--prompt-templates-configs`"""
	)
	parser.add_argument(
		"--cache-dir",
		type=str,
		required=True,
		help="Path to the cache dir of huggingface datasets. (The directory may require very large space.)",
	)
	parser.add_argument(
		"--output-dir", type=str, required=True, 
		help="Path to the output dir where the projected data will be stored."
	)
	parser.add_argument(
		"--num-proc",
		type=int,
		default=9,
		help="Total number of parallel process."
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
	args = parser.parse_args()

	assert len(args.dataset_name_or_paths) == len(args.dataset_configs)
	assert len(args.dataset_name_or_paths) == len(args.prompt_templates_configs)

	invoke_none(args.dataset_name_or_paths)
	invoke_none(args.dataset_configs)
	invoke_none(args.prompt_templates_configs)

	prompted_sample_gen_io_tuple_list = []
	# loading and caching each of the dataset & creating multiprocessor i/o for doing projection.
	for (dataset_name_or_path, dataset_config, prompt_template_config) in zip(
		args.dataset_name_or_paths, args.dataset_configs, args.prompt_templates_configs
	):
		dataset = datasets.load_dataset(dataset_name_or_path, dataset_config, cache_dir=args.cache_dir)
		psrc_prompt_template_signature = prompt_template_config
		if psrc_prompt_template_signature is None:
			if dataset_config is None:
				psrc_prompt_template_signature = "{}".format(dataset_name_or_path)
			else:
				psrc_prompt_template_signature = "{}/{}".format(dataset_name_or_path, dataset_config)
		dataset_output_dir = os.path.join(args.output_dir, dataset_name_or_path)
		os.makedirs(dataset_output_dir, exist_ok=True)
		if dataset_config is not None:
			dataset_output_dir = os.path.join(dataset_output_dir, dataset_config)
			os.makedirs(dataset_output_dir, exist_ok=True)
		prompt_templates = DatasetTemplates(psrc_prompt_template_signature)
		prompt_names = list(prompt_templates.name_to_id_mapping.keys())
		for prompt_name in prompt_names:
			prompt_template = prompt_templates[prompt_name]
			prompted_sample_gen_io_tuple = (dataset_output_dir,
											dataset_name_or_path,
											dataset_config,
											psrc_prompt_template_signature,
											prompt_template,
											dataset,
											args.add_source_metadata,
											args.highlight_variables)
			prompted_sample_gen_io_tuple_list.append(prompted_sample_gen_io_tuple)
	
	# Test a single process run
	# export_dataset(
	# 	prompted_sample_gen_io_tuple_list[0][0],
	# 	prompted_sample_gen_io_tuple_list[0][1],
	# 	prompted_sample_gen_io_tuple_list[0][2],
	# 	prompted_sample_gen_io_tuple_list[0][3],
	# 	prompted_sample_gen_io_tuple_list[0][4],
	# 	prompted_sample_gen_io_tuple_list[0][5],
	# 	prompted_sample_gen_io_tuple_list[0][6],
	# 	prompted_sample_gen_io_tuple_list[0][7],
	# )

	# Projecting data using multiprocessing. It's recommended to use large number of CPU machine. set up `--num-proc` accrodingly. 
	num_proc = min(args.num_proc, len(prompted_sample_gen_io_tuple_list))

	with concurrent.futures.ProcessPoolExecutor(
		max_workers=num_proc
	) as executor:
		for _out in tqdm(
			executor.map(
				export_dataset,
				[prompted_sample_gen_io[0] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # dataset_output_dir
			[prompted_sample_gen_io[1] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # dataset_name_or_path
			[prompted_sample_gen_io[2] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # dataset_config
			[prompted_sample_gen_io[3] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # psrc_prompt_template_signature
			[prompted_sample_gen_io[4] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # prompt_template
			[prompted_sample_gen_io[5] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # dataset
			[prompted_sample_gen_io[6] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # args.add_source_metadata
			[prompted_sample_gen_io[7] for prompted_sample_gen_io in prompted_sample_gen_io_tuple_list], # args.highlight_variables
			),
			total=len(args.dataset_name_or_paths),
		):
			try:
				print(_out)
			except Exception as emsg:
				print("Exception msg: {}".format(emsg))

if __name__ == "__main__":
	main()