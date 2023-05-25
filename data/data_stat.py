import os
import csv
import json
import datasets
import argparse

# huggingface dataset signature with configs
SERIES_A_DATASET_NAME_DICT = {
	"udhr": {
		None: "mixed"
	},
	"AmazonScience/mintaka": {
		"ar": "ar",
		"de": "de",
		"en": "en",
		"es": "es", 
		"fr": "fr",
		"hi": "hi",
		"it": "it",
		"ja": "ja",
		"pt": "pt",
	},
	"xcsr": {
		'X-CSQA-en': "en", 
		'X-CSQA-zh': "zh", 
		'X-CSQA-de': "de", 
		'X-CSQA-es': "es", 
		'X-CSQA-fr': "fr", 
		'X-CSQA-it': "it", 
		'X-CSQA-jap': "ja", 
		'X-CSQA-nl': "nl", 
		'X-CSQA-pl': "pl", 
		'X-CSQA-pt': "pt", 
		'X-CSQA-ru': "ru", 
		'X-CSQA-ar': "ar", 
		'X-CSQA-vi': "vi", 
		'X-CSQA-hi': "hi", 
		'X-CSQA-sw': "sw", 
		'X-CSQA-ur': "ur", 
		# 'X-CODAH-en': "en", 
		# 'X-CODAH-zh': "zh", 
		# 'X-CODAH-de': "de", 
		# 'X-CODAH-es': "es", 
		# 'X-CODAH-fr': "fr", 
		# 'X-CODAH-it': "it", 
		# 'X-CODAH-jap': "ja", 
		# 'X-CODAH-nl': "nl", 
		# 'X-CODAH-pl': "pl", 
		# 'X-CODAH-pt': "pt", 
		# 'X-CODAH-ru': "ru", 
		# 'X-CODAH-ar': "ar", 
		# 'X-CODAH-vi': "vi", 
		# 'X-CODAH-hi': "hi", 
		# 'X-CODAH-sw': "sw", 
		# 'X-CODAH-ur': "ur",
	},
	"shmuhammad/AfriSenti-twitter-sentiment": {
		'amh':'amh', 
		'hau':'hau', 
		'ibo':'ibo',
		'arq':'arq', 
		'ary':'ary', 
		# 'yor':'yor', 
		'por':'por', 
		'twi':'twi', 
		'tso':'tso', 
		'tir':'tir', 
		'pcm':'pcm', 
		'kin':'kin', 
		'swa': 'swa',
		# 'orm': 'orm',
	}, 
	"indonlp/NusaX-senti": {
		'ace':'ace', 
		'ban':'ban', 
		'bjn':'bjn', 
		# 'bug':'bug',
		'eng':'eng',
		'ind':'ind',
		# 'jav':'jav', 
		'mad':'mad', 
		'min':'min', 
		'nij':'nij', 
		'sun':'sun', 
		'bbc':'bbc',
	},
	"masakhane/masakhanews": {
		'amh':'amh', 
		'eng':'eng', 
		'fra':'fra', 
		'hau':'hau', 
		'ibo':'ibo', 
		'lin':'lin', 
		'lug':'lug', 
		'orm':'orm', 
		'pcm':'pcm', 
		'run':'run', 
		'sna':'sna', 
		'som':'som', 
		'swa':'swa', 
		'tir':'tir', 
		'xho':'xho', 
		'yor':'yor',	
	},
	"papluca/language-identification": {
		None: "mixed",
	},
	"adithya7/xlel_wd": {
		'wikipedia-zero-shot': "mixed",
		'wikinews-zero-shot': "mixed",
		'wikinews-cross-domain': "mixed", 
		'wikipedia-zero-shot.af': 'af', 
		'wikipedia-zero-shot.ar': 'ar', 
		'wikipedia-zero-shot.be': 'be', 
		'wikipedia-zero-shot.bg': 'bg', 
		'wikipedia-zero-shot.bn': 'bn',
		'wikipedia-zero-shot.ca': 'ca',
		'wikipedia-zero-shot.cs': 'cs', 
		'wikipedia-zero-shot.da': 'da', 
		'wikipedia-zero-shot.de': 'de', 
		'wikipedia-zero-shot.el': 'el', 
		'wikipedia-zero-shot.en': 'en',
		'wikipedia-zero-shot.es': 'es',
		'wikipedia-zero-shot.fa': 'fa', 
		'wikipedia-zero-shot.fi': 'fi',
		'wikipedia-zero-shot.fr': 'fr',
		'wikipedia-zero-shot.he': 'he',
		'wikipedia-zero-shot.hi': 'hi',
		'wikipedia-zero-shot.hu': 'hu',
		'wikipedia-zero-shot.id': 'id',
		'wikipedia-zero-shot.it': 'it',
		'wikipedia-zero-shot.ja': 'ja',
		'wikipedia-zero-shot.ko': 'ko',
		'wikipedia-zero-shot.ml': 'ml',
		'wikipedia-zero-shot.mr': 'mr',
		'wikipedia-zero-shot.ms': 'ms',
		'wikipedia-zero-shot.nl': 'nl',
		'wikipedia-zero-shot.no': 'no',
		'wikipedia-zero-shot.pl': 'pl',
		'wikipedia-zero-shot.pt': 'pt',
		'wikipedia-zero-shot.ro': 'ro',
		'wikipedia-zero-shot.ru': 'ru',
		'wikipedia-zero-shot.si': 'si',
		'wikipedia-zero-shot.sk': 'sk',
		'wikipedia-zero-shot.sl': 'sl',
		'wikipedia-zero-shot.sr': 'sr', 
		'wikipedia-zero-shot.sv': 'sv', 
		'wikipedia-zero-shot.sw': 'sw',
		'wikipedia-zero-shot.ta': 'ta', 
		'wikipedia-zero-shot.te': 'te',
		'wikipedia-zero-shot.th': 'th',
		'wikipedia-zero-shot.tr': 'tr',
		'wikipedia-zero-shot.uk': 'uk',
		'wikipedia-zero-shot.vi': 'vi',
		'wikipedia-zero-shot.zh': 'zh',
		'wikinews-zero-shot.ar': 'ar',
		'wikinews-zero-shot.cs': 'cs',
		'wikinews-zero-shot.de': 'de',
		'wikinews-zero-shot.en': 'en',
		'wikinews-zero-shot.es': 'es',
		'wikinews-zero-shot.fi': 'fi', 
		'wikinews-zero-shot.fr': 'fr',
		'wikinews-zero-shot.it': 'it',
		'wikinews-zero-shot.ja': 'ja',
		'wikinews-zero-shot.ko': 'ko',
		'wikinews-zero-shot.nl': 'nl',
		'wikinews-zero-shot.no': 'no',
		'wikinews-zero-shot.pl': 'pl',
		'wikinews-zero-shot.pt': 'pt',
		'wikinews-zero-shot.ru': 'ru',
		'wikinews-zero-shot.sr': 'sr',
		'wikinews-zero-shot.sv': 'sv',
		'wikinews-zero-shot.ta': 'ta',
		# 'wikinews-zero-shot.tr': 'tr',
		'wikinews-zero-shot.uk': 'uk',
		'wikinews-zero-shot.zh': 'zh',
		'wikinews-cross-domain.ar': 'ar',
		'wikinews-cross-domain.bg': 'bg',
		'wikinews-cross-domain.ca': 'ca',
		'wikinews-cross-domain.cs': 'cs',
		'wikinews-cross-domain.de': 'de',
		'wikinews-cross-domain.el': 'el',
		'wikinews-cross-domain.en': 'en',
		'wikinews-cross-domain.es': 'es',
		'wikinews-cross-domain.fi': 'fi',
		'wikinews-cross-domain.fr': 'fr',
		'wikinews-cross-domain.he': 'he', 
		'wikinews-cross-domain.hu': 'hu', 
		'wikinews-cross-domain.it': 'it', 
		'wikinews-cross-domain.ja': 'ja',
		'wikinews-cross-domain.ko': 'ko',
		'wikinews-cross-domain.nl': 'nl',
		'wikinews-cross-domain.no': 'no',
		'wikinews-cross-domain.pl': 'pl',
		'wikinews-cross-domain.pt': 'pt',
		'wikinews-cross-domain.ro': 'ro',
		'wikinews-cross-domain.ru': 'ru',
		'wikinews-cross-domain.sr': 'sr',
		'wikinews-cross-domain.sv': 'sv',
		'wikinews-cross-domain.ta': 'ta',
		'wikinews-cross-domain.tr': 'tr',
		'wikinews-cross-domain.uk': 'uk', 
		'wikinews-cross-domain.zh': 'zh',
	},
	"sbmaruf/forai_ml-ted_talk_iwslt": {
		'eu_ca_2014': 'eu_ca', 
		'eu_ca_2015': 'eu_ca', 
		'eu_ca_2016': 'eu_ca', 
		'nl_en_2014': 'nl_en', 
		'nl_en_2015': 'nl_en', 
		'nl_en_2016': 'nl_en', 
		'nl_hi_2014': 'nl_hi', 
		'nl_hi_2015': 'nl_hi', 
		'nl_hi_2016': 'nl_hi', 
		'de_ja_2014': 'de_ja', 
		'de_ja_2015': 'de_ja', 
		'de_ja_2016': 'de_ja', 
		'fr-ca_hi_2014': 'fr_hi', 
		'fr-ca_hi_2015': 'fr_hi', 
		'fr-ca_hi_2016': 'fr_hi',
	},
	"sbmaruf/forai_ml_masakhane_mafand":{
		'en-amh': 'en-amh', 
		'en-hau': 'en-hau', 
		'en-ibo': 'en-ibo', 
		'en-kin': 'en-kin', 
		'en-lug': 'en-lug', 
		'en-nya': 'en-nya', 
		'en-pcm': 'en-pcm', 
		'en-sna': 'en-sna', 
		'en-swa': 'en-swa', 
		'en-tsn': 'en-tsn', 
		'en-twi': 'en-twi', 
		'en-xho': 'en-xho', 
		'en-yor': 'en-yor', 
		'en-zul': 'en-zul', 
		'fr-bam': 'fr-bam', 
		'fr-bbj': 'fr-bbj', 
		'fr-ewe': 'fr-ewe', 
		'fr-fon': 'fr-fon', 
		'fr-mos': 'fr-mos', 
		'fr-wol': 'fr-wol',
	},
	"exams":{
		# 'alignments': 'mixed', 
		'multilingual': 'mixed', 
		'multilingual_with_para': 'mixed', 
		'crosslingual_test':'mixed', 
		'crosslingual_with_para_test': 'mixed', 
		'crosslingual_bg': "bg", 
		'crosslingual_with_para_bg': "bg", 
		'crosslingual_hr': "hr", 
		'crosslingual_with_para_hr': "hr", 
		'crosslingual_hu': "hu", 
		'crosslingual_with_para_hu': "hu", 
		'crosslingual_it': "it", 
		'crosslingual_with_para_it': "it", 
		'crosslingual_mk': "mk", 
		'crosslingual_with_para_mk': "mk", 
		'crosslingual_pl': "pl", 
		'crosslingual_with_para_pl': "pl", 
		'crosslingual_pt': "pt", 
		'crosslingual_with_para_pt': "pt", 
		'crosslingual_sq': "sq", 
		'crosslingual_with_para_sq': "sq", 
		'crosslingual_sr': "sr", 
		'crosslingual_with_para_sr': "sr", 
		'crosslingual_tr': "tr", 
		'crosslingual_with_para_tr': "tr", 
		'crosslingual_vi': "vi", 
		'crosslingual_with_para_vi': "vi",
	},
	"allenai/soda": {
		None: "en",
	}, 
	"arabic_billion_words": {
		'Alittihad': "Alittihad", 
		'Almasryalyoum': "Almasryalyoum", 
		'Almustaqbal': "Almustaqbal", 
		'Alqabas': "Alqabas", 
		'Echoroukonline': "Echoroukonline", 
		'Ryiadh': "Ryiadh", 
		'Sabanews': "Sabanews", 
		'SaudiYoumSaudi': "", 
		'Techreen': "Techreen", 
		'Youm7': "Youm7",
	},
	"theblackcat102/joke_explaination": {
		None: "en",
	},
	"narrativeqa": {
		None: "en",
	},
	"svakulenk0/qrecc": {
		None: "en",
	},
	"GEM/wiki_cat_sum": {
		"animan": "en",
		"company": "en",
		"film": "en",
	}
}

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset-names",
		nargs="+",
		default=None,
		help="Print the stat of the dataset. If `None` it will print stat of all the used data."
	)
	parser.add_argument(
		"--export-format",
		choices=['json', "csv"],
		default=".json",
		help="Which format you want to export."
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="The path to the folder where stat will be saved."
	)
	args = parser.parse_args()
	stat_dict = {}
	if args.dataset_names is None:
		args.dataset_names = list(SERIES_A_DATASET_NAME_DICT.keys())
	for dataset_name, subset_dict in SERIES_A_DATASET_NAME_DICT.items():
		if dataset_name not in args.dataset_names:
			continue
		assert dataset_name not in stat_dict
		stat_dict[dataset_name] = {}
		for subset, subset_lang in subset_dict.items():
			assert subset not in stat_dict[dataset_name]
			stat_dict[dataset_name][subset] = {}
			dt = datasets.load_dataset(dataset_name, name=subset, verification_mode="no_checks")
			for split in dt.keys():
				stat_dict[dataset_name][subset][split] = {
					"size": len(dt[split]),
					"column": list(dt[split].column_names),
				}
				# re-valuation of hypothesis considered in prompt template
				if subset is not None and "X-CSQA" in subset:
					for sample in dt[split]:
						assert len(sample['question']['choices']['label']) == 5

	if args.output_dir != 'None': 
		file_name = os.path.join(args.output_dir, "stat") + f".{args.export_format}"
		if args.export_format == "json":
			with open(file_name, "w") as file_ptr:
				file_ptr.write(f"{json.dumps(stat_dict, indent=4)}\n")
		elif args.export_format == "csv":
			# with open(file_name, mode='w') as file_ptr:
			# 	writer = csv.writer(file_ptr)
			# 	for dataset_name, subset_name, in SERIES_A_DATASET_NAME_DICT.keys():
			# 		row = [f"{dataset_name}"]
				
			# 	writer.writerow(stat_dict.values())
			pass
		else:
			raise NotImplementedError

if __name__ == "__main__":
	main()
   
