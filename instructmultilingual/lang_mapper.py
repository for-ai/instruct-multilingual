import json
import logging

from flores_200 import lang_code_to_name as nllb_lang_code_to_name
from iso639 import languages as iso_languages

# Taken from https://arxiv.org/pdf/2010.11934.pdf table 6
# https://github.com/google-research/multilingual-t5#languages-covered
mt5_langs_name_pair = [("Afrikaans", "af"), ("Albanian", "sq"), ("Amharic", "am"), ("Arabic", "ar"), ("Armenian", "hy"),
                       ("Azerbaijani", "az"), ("Basque", "eu"), ("Belarusian", "be"), ("Bengali", "bn"),
                       ("Bulgarian", "bg"), ("Burmese", "my"), ("Catalan", "ca"), ("Cebuano", "ceb"),
                       ("Chichewa", "ny"), ("Chinese", "zh"), ("Corsican", "co"), ("Czech", "cs"), ("Danish", "da"),
                       ("Dutch", "nl"), ("English", "en"), ("Esperanto", "eo"), ("Estonian", "et"), ("Filipino", "fil"),
                       ("Finnish", "fi"), ("French", "fr"), ("Galician", "gl"), ("Georgian", "ka"), ("German", "de"),
                       ("Greek", "el"), ("Gujarati", "gu"), ("Haitian Creole", "ht"), ("Hausa", "ha"),
                       ("Hawaiian", "haw"), ("Hebrew", "iw"), ("Hindi", "hi"), ("Hmong", "hmn"), ("Hungarian", "hu"),
                       ("Icelandic", "is"), ("Igbo", "ig"), ("Indonesian", "id"), ("Irish", "ga"), ("Italian", "it"),
                       ("Japanese", "ja"), ("Javanese", "jv"), ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"),
                       ("Korean", "ko"), ("Kurdish", "ku"), ("Kyrgyz", "ky"), ("Lao", "lo"), ("Latin", "la"),
                       ("Latvian", "lv"), ("Lithuanian", "lt"), ("Luxembourgish", "lb"), ("Macedonian", "mk"),
                       ("Malagasy", "mg"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"),
                       ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"), ("Norwegian", "no"), ("Pashto", "ps"),
                       ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"),
                       ("Russian", "ru"), ("Samoan", "sm"), ("Scottish Gaelic", "gd"), ("Serbian", "sr"),
                       ("Shona", "sn"), ("Sindhi", "sd"), ("Sinhala", "si"), ("Slovak", "sk"), ("Slovenian", "sl"),
                       ("Somali", "so"), ("Sotho", "st"), ("Spanish", "es"), ("Sundanese", "su"), ("Swahili", "sw"),
                       ("Swedish", "sv"), ("Tajik", "tg"), ("Tamil", "ta"), ("Telugu", "te"), ("Thai", "th"),
                       ("Turkish", "tr"), ("Ukrainian", "uk"), ("Urdu", "ur"), ("Uzbek", "uz"), ("Vietnamese", "vi"),
                       ("Welsh", "cy"), ("West Frisian", "fy"), ("Xhosa", "xh"), ("Yiddish", "yi"), ("Yoruba", "yo"),
                       ("Zulu", "zu")]


def test_iso_validity(ISO_LANG_NAME_LIST):
    """
	1. Test if all mT5 languages are in ISO list or not.
	2. Test if there is any mismatch in naming between mT5 and ISO.    
	"""
    for full_lang, short_lang in mt5_langs_name_pair:
        if full_lang not in ISO_LANG_NAME_LIST:
            logging.warning(f"[*****] lang {full_lang} not in ISO-639")
        else:
            full_lang_from_lib = iso_languages.get(name=full_lang)
            if full_lang_from_lib.part1 == short_lang:
                continue
            elif full_lang_from_lib.part3 == short_lang:
                continue
            elif full_lang_from_lib.part5 == short_lang:
                continue
            elif full_lang_from_lib.part2t == short_lang:
                continue
            elif full_lang_from_lib.part2b == short_lang:
                continue
            else:
                logging.warning(
                    f"name: {full_lang_from_lib.name}, part1: {full_lang_from_lib.part1}, part2b: {full_lang_from_lib.part2b}, part2t: {full_lang_from_lib.part2t}, part5: {full_lang_from_lib.part5}, mt5_sign: {short_lang}"
                )


def get_mt5_2_nllb_mapper(mt5_langs_name_pair, nllb_lang_names):
    """Create mapper between mT5 to nllb languages."""
    MT5_2_NLLB = {}
    for full_lang, short_lang in mt5_langs_name_pair:
        names_found = []
        for nllb_lang_name, nllb_lang_short_nname in nllb_lang_names:
            if full_lang in nllb_lang_name:
                names_found.append((nllb_lang_short_nname, nllb_lang_name))
        if len(names_found) == 0:
            logging.warning(f"[-----] {full_lang} not found in NLLB.")
        else:
            MT5_2_NLLB[short_lang] = names_found
    return MT5_2_NLLB


def main():
    mt5_full_lang_names = set([full_lang for (full_lang, _) in mt5_langs_name_pair])
    assert len(mt5_full_lang_names) == 101
    mt5_short_lang_names = set([short_lang for (_, short_lang) in mt5_langs_name_pair])
    assert len(mt5_short_lang_names) == 101

    nllb_lang_names = [(v, k) for k, v in nllb_lang_code_to_name.items()]
    ISO_LANG_NAME_LIST = [lag_obj.name for lag_obj in list(iso_languages)]

    test_iso_validity(ISO_LANG_NAME_LIST)
    MT5_2_NLLB = get_mt5_2_nllb_mapper(mt5_langs_name_pair, nllb_lang_names)
    # print(json.dumps(MT5_2_NLLB, indent=4))
    with open("MT5_2_NLLB.json", "w") as file_ptr:
        file_ptr.write(json.dumps(MT5_2_NLLB, indent=4))


if __name__ == "__main__":
    main()
