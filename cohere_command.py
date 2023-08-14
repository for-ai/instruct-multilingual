import os
import cohere
import pandas as pd

from tqdm import tqdm

co = cohere.Client(os.environ["COHERE_API_KEY"])

# Need to download files from here since there is a bug in huggingface's datasets library
# - https://huggingface.co/datasets/RyokoAI/ShareGPT52K/blob/main/sg_90k_part1.json
# - https://huggingface.co/datasets/RyokoAI/ShareGPT52K/blob/main/sg_90k_part2.json
dataset = pd.read_json("gs://cohere-dev-central-1/amr/aya/sg_90k_part1.json")

output = {
    "id": [],
    "prompt": [],
    "completion": [],
}

pbar = tqdm(range(len(dataset)), total=len(dataset), desc="Dataset Examples", position=0)
for conv_id in pbar:

    example = dataset.iloc[conv_id]
    conversation = example['conversations']

    chat_history = []
    max_turns = len(conversation) // 2
    conversation_id = None

    pbar_conv = tqdm(range(max_turns), total=max_turns, desc="Conversation Turns", position=1)
    for i in pbar_conv:
        prompt = conversation[i * 2]['value']
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=300,
            temperature=0.9,
        )

        completion = response.generations[0].text

        output["id"].append(example['id'])
        output["prompt"].append(prompt)
        output["completion"].append(completion)

# import ipdb; ipdb.set_trace()
output = pd.DataFrame(output)
output.to_json("./command_prompt_completion.json")
