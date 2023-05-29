import os
import csv
import json


def read_file_paths(file_path):
    file_paths = set({})
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for _idx, row in enumerate(csv_reader):
            if _idx == 0:
                continue
            file_paths.add(row[-1])
    return file_paths


def list_files_recursively(folder_path):
    path_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            path_list.append(file_path)
    return path_list


TASKS = [
    # "AmazonScience_mintaka",
    # "GEM_wiki_cat_sum",
    # "exams",
    # "adithya7_xlel_wd",
    # "wiki_split",
    # "xcsr",
    # "TurkuNLP_turku_paraphrase_corpus",
    # "theblackcat102_joke_explaination",
    # "allenai_scirepeval",
    # "soda",
]
for task in TASKS:
    files = sorted(list_files_recursively(f"./dumped_{task}/"))
    prompted_sample_file_path = f"dumped_{task}/single_sample_{task}.csv"
    file_paths_in_prompted_sample = {}
    FLAG = "w"
    if os.path.exists(prompted_sample_file_path):
        file_paths_in_prompted_sample = read_file_paths(prompted_sample_file_path)
        FLAG = "a"

    with open(prompted_sample_file_path, FLAG, newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        FLAG = 1
        rows = []
        for _idx, _file in enumerate(files):
            if _file.endswith("jsonl"):
                print(_idx, _file)
                data = json.loads(next(iter(open(_file))))
                if FLAG:
                    HEADERS = list(data.keys())
                    HEADERS.append(["FILE PATH"])
                    csv_writer.writerow(HEADERS)
                    FLAG = 0
                row = [data[header] for header in HEADERS[:-1]]
                row.append(_file)
                rows.append(row)
        print(f"Total rows {len(rows)}")
        for row in rows:
            if row[-1] not in file_paths_in_prompted_sample:
                csv_writer.writerow(row)
