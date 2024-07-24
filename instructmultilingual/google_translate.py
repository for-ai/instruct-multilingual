from os import environ
import pandas as pd
from google.cloud import translate, translate_v2


PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"


def print_supported_languages(display_language_code: str):
    client = translate.TranslationServiceClient()

    response = client.get_supported_languages(
        parent=PARENT,
        display_language_code=display_language_code,
    )

    languages = response.languages
    print(f" Languages: {len(languages)} ".center(60, "-"))
    for language in languages:
        language_code = language.language_code
        display_name = language.display_name
        print(f"{language_code:10}{display_name}")


# print_supported_languages("en")

def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=text,
        target_language_code=target_language_code,
        mime_type="text/plain"
    )

    return response #.translations[0]

def translate_basic(text, target_lang_code):
    translate_client = translate_v2.Client()
    results = translate_client.translate(text, target_language=target_lang_code)
    return results["translatedText"]
    
text = ['The most important element of ethical research on human subjects is:', 'The so-called “bigfoot” on Mars was actually a rock that was about 5 cm tall. It had an angular size of about 0.5 degrees (~30 pixels). How far away was this rock from the rover?', 'Post-modern ethics assert that ethics are context and individual specific and as such have an internal guide to ethics.', 'Which of the following statements are correct concerning the use of antithetic variates as part of a Monte Carlo experiment?\n\ni) Antithetic variates work by reducing the number of replications required to cover the whole probability space\n\nii) Antithetic variates involve employing a similar variable to that used in the simulation, but whose properties are known analytically\n\niii) Antithetic variates involve using the negative of each of the random draws and repeating the experiment using those values as the draws\n\niv) Antithetic variates involve taking one over each of the random draws and repeating the experiment using those values as the draws']


target_languages = ["hi", "de", "es", "it", "el", "zh", "ja", "ko"]
print(f" {text} ".center(50, "-"))
for target_language in target_languages:
    responses = translate_text(text, target_language)
    # translation_basic = translate_basic(text, target_language)
    # print(translation)
    translations=[translation.translated_text for translation in responses.translations]
    print(translations)
    # source_language = translation.detected_language_code
    # translated_text = translation.translated_text
    # print(f"{source_language} → {target_language} : {translated_text}")
    break
    # print("translation basic: --> ", translation_basic)