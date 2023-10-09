import itertools
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import pandas as pd
import sentencepiece as spm


def get_token_count(row, sp):
    tok_count_input = len(sp.encode(row[1][0]))
    tok_count_target = len(sp.encode(row[1][1]))
    return tok_count_input, tok_count_target


def process_df(root, f, sp):

    f_start_t = time.time()

    file_info = root.split(os.path.sep)
    dataset_name, translation, translation_model, template_name, date = file_info[-5], file_info[-4], file_info[
        -3], file_info[-2], file_info[-1]

    return_dict = {
        "dataset_name": dataset_name,
        "translation": translation,
        "translation_model": translation_model,
        "template_name": template_name,
        "date": date,
        "source_file": f,
        "token_count_inputs": 0,
        "token_count_targets": 0,
        "unique_token_count_inputs": {},
        "unique_token_count_targets": {}
    }

    try:
        filepath = os.path.join(root, f)
        data_df = pd.read_csv(filepath).dropna()
        filestats = os.stat(filepath)
        

        tok_inputs = [sp.encode(text) for text in data_df["inputs"]]
        tok_targets = [sp.encode(text) for text in data_df["targets"]]


        data_df["token_count_inputs"] = [len(toks) for toks in tok_inputs]
        data_df["token_count_targets"] = [len(toks) for toks in tok_targets]

        total_tok_inputs = list(itertools.chain.from_iterable(tok_inputs))
        total_tok_targets = list(itertools.chain.from_iterable(tok_targets))


    except Exception as e:
        print(f"ERROR: {e}: {filepath}", flush=True)
        return return_dict

    print(f"{filepath}: {time.time()-f_start_t:.2f}", flush=True)

    return_dict["filesize"] = filestats.st_size
    return_dict["token_count_inputs"] = data_df["token_count_inputs"].sum()
    return_dict["token_count_targets"] = data_df["token_count_targets"].sum()
    return_dict["unique_token_count_inputs"] = dict(Counter(total_tok_inputs))
    return_dict["unique_token_count_targets"] = dict(Counter(total_tok_targets))
    del data_df
    return return_dict


def main(source_pth: str,
         output_pth: str,
         output_name: str = 'data_stats.csv',
         tokenizer_model_pth: str = 'sentencepiece.model',
         workers: int = os.cpu_count()) -> None:

    start_t = time.time()
    key_names = ["dataset_name","template_name","date","translation_model","translation","source_file","token_count_inputs",
                 "token_count_targets","unique_token_count_inputs","unique_token_count_targets", "filesize"]

    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_pth)


    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []

        for root, _, files in os.walk(source_pth):
            if len(files) > 0:
                for f in files:
                    if f.endswith(".csv"):
                        futures.append(executor.submit(process_df, root, f, sp))

        with open(os.path.join(output_pth, output_name), 'w') as o:
        
            header_str = ",".join(key_names)
            header_str += '\n'
            o.write(header_str)

            for future in as_completed(futures):
                return_dict = future.result()
                str_reps = []
                for k in key_names:
                    if "unique" in k:
                        str_reps.append(f'"{return_dict[k]}"') 
                    else:
                        str_reps.append(str(return_dict[k]))
                line = ",".join(str_reps)
                line += '\n'
                o.write(line)




    print(f"{source_pth} total_runtime: {time.time()-start_t:.2f}")


if __name__ == '__main__':
    fire.Fire(main)
