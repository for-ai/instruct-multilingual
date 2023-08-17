import os
from multiprocessing.pool import ThreadPool
from typing import Tuple

import cohere
import pandas as pd
from pyrate_limiter import Duration, Limiter, MemoryListBucket, RequestRate
from tqdm import tqdm

minute_rate = RequestRate(10000, Duration.MINUTE)
limiter = Limiter(minute_rate, bucket_class=MemoryListBucket)
NUM_THREADS = 2 * os.cpu_count()  # 2 threads per cpu core is standard

co = cohere.Client(os.environ["COHERE_API_KEY"])

# Need to download files from here since there is a bug in huggingface's datasets library
# - https://huggingface.co/datasets/RyokoAI/ShareGPT52K/blob/main/sg_90k_part1.json
# - https://huggingface.co/datasets/RyokoAI/ShareGPT52K/blob/main/sg_90k_part2.json
dataset = pd.read_json("gs://cohere-dev-central-1/amr/aya/sg_90k_part2.json")


@limiter.ratelimit("blobheart", delay=True)
def api_generation(inputs: Tuple[cohere.Client, str]):
    co, id, prompt = inputs
    try:
        generation = co.generate(
            model='command-nightly',
            prompt=prompt,
            max_tokens=300,
            temperature=0.9,
            truncate='END',
        ).generations[0]
    except cohere.CohereError as exception:
        print(exception)
        return {
            'id': id,
            'prompt': prompt,
            'completion': f'BLOBHEART_EXCEPTION_ERROR_{exception}',
        }
    return {
        'id': id,
        'prompt': prompt,
        'completion': generation.text,
    }


output = {
    "id": [],
    "prompt": [],
    "completion": [],
}

pbar = tqdm(range(len(dataset)), total=len(dataset), desc="Dataset Examples", position=0)
prompts = []
for conv_id in pbar:

    example = dataset.iloc[conv_id]
    conversation = example['conversations']

    chat_history = []
    max_turns = len(conversation) // 2
    conversation_id = None

    for i in range(max_turns):
        prompt = conversation[i * 2]['value']
        prompts.append((co, example['id'], prompt.strip()))

print(f"Total number of prompts: {len(prompts)}")
with ThreadPool(NUM_THREADS) as pool:
    generations = list(tqdm(pool.imap(api_generation, prompts), total=len(prompts)))

# import ipdb; ipdb.set_trace()
for generation in generations:
    output["id"].append(generation['id'])
    output["prompt"].append(generation['prompt'])
    output["completion"].append(generation['completion'])

# import ipdb; ipdb.set_trace()
output = pd.DataFrame(output)
output.to_json("./command_prompt_completion_part2.json")
