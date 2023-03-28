import requests
import time
from datasets import load_dataset


url = "http://localhost:8000/translate"
headers = {"Content-Type": "application/json"}


def translate(source_language, target_language, texts):
    headers = {"Content-Type": "application/json"}
    data = {
        "source_language": source_language,
        "target_language": target_language,
        "texts": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["translated_texts"]


def tokenization(example, keys_to_be_translated=["dialogue", "summary"]):
    for key in keys_to_be_translated:
        example[key] = translate("eng_Latn", "arz_Arab", example[key])
    return example


start_time = time.time()

dataset = load_dataset("samsum")
for split in ["train", "validation", "test"]:
    split_time = time.time()
    ds = dataset[split]
    print(f"[{split}] {len(ds)=}")
    ds = ds.map(tokenization, batched=True, num_proc=8)
    print(f"[{split}] One example translated {ds[0]=}")
    print(f"[{split}] took {time.time() - split_time:.4f} seconds")

end_time = time.time()
elapsed_time = end_time - start_time


print(f"Elapsed time: {elapsed_time:.4f} seconds")
