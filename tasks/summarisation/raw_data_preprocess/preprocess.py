import os
import csv
import json
from tqdm import tqdm
from pathlib import Path
import sys
import glob
import os
from pathlib import Path
import numpy as np
import jsonlines

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.file_utils import load_json, save_json, pickle_save, save_jsonl
from src.utils.summarisation.data_utils import extract_doc_content
from copy import deepcopy

np.random.seed(42)

class RawDataObject(object):
    def __init__(
        self,
        cord_uid = None,
        sha = None,
        source_x  = None,
        title_ = None,
        doi  = None,
        pmcid = None,
        pubmed_id  = None,
        license = None,
        abstract = None,
        publish_time = None,
        authors = None,
        journal = None,
        mag_id  = None,
        who_covidence_id = None,
        arxiv_id = None,
        pdf_json_files = None,
        pmc_json_files = None,
        url = None,
        s2_id = None,
    ):
        self.cord_uid = cord_uid
        self.sha = sha
        self.source_x  = source_x
        self.title_ = title_
        self.doi  = doi
        self.pmcid = pmcid
        self.pubmed_id  = pubmed_id
        self.license = license
        self.abstract = abstract
        self.publish_time = publish_time
        self.authors = authors
        self.journal = journal
        self.mag_id  = mag_id
        self.who_covidence_id = who_covidence_id
        self.arxiv_id = arxiv_id
        self.pdf_json_files = pdf_json_files
        self.pmc_json_files = pmc_json_files
        self.url = url
        self.s2_id = s2_id
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class CovidDataProcessor(object):
    def __init__(self, raw_data_dir: Path, output_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.id_pdf_path = output_dir.joinpath('id_pdf_dict.json')
        self.covid_data_path: Path = self.output_dir.joinpath('covid_data.json')

        self.minimal_ref_num = 3

    def has_missing_attr(self, paper: dict):
        for key, val in paper.items():
            if val == [] or val == '':
                return True
        return False

    def make_covid_data(self):
        raw_data = self.read_raw_covid_corpus(self.raw_data_dir)

        id_pdf_dic = {}
        for sam in raw_data:
            id_pdf_dic[sam.title_] = {'id': sam.pubmed_id, 'files': sam.pdf_json_files}
        save_json(id_pdf_dic, self.id_pdf_path)

        refined_covid_json = []
        for sample in tqdm(raw_data, "make covid data"):
            paper = extract_doc_content(sample.pdf_json_files, id_pdf_dic)
            if paper:
                paper['abstract'] = sample.abstract
                paper['title'] = sample.title_
                paper['pdf'] = sample.pdf_json_files
                paper['paper_id'] = sample.pubmed_id
                if self.has_missing_attr(paper):
                    continue
                paper["ref_list"] = deepcopy(paper["references"])
                paper["references"] = list(set(paper["references"]))
                if len(paper["references"]) >= self.minimal_ref_num:
                    refined_covid_json.append(paper)
        save_json(refined_covid_json, self.covid_data_path)
        print(f"write json file to {self.covid_data_path}; size: {len(refined_covid_json)}")

    def _split_data(self, data, test_ratio=0.1, val_ratio=0.1, data_limit=100000):
        data = data[:data_limit]

        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        val_set_size = int(len(data) * val_ratio)
        test_indices = shuffled_indices[:test_set_size]
        val_indices = shuffled_indices[test_set_size:test_set_size+val_set_size]
        train_indices = shuffled_indices[test_set_size+val_set_size:]

        return [data[i] for i in train_indices], [data[i] for i in test_indices], [data[i] for i in val_indices]

    def clean_covid_data(self, covid_data: list):
        paper_ids = set([paper['paper_id'] for paper in covid_data])
        new_covid_data = []
        for paper in tqdm(covid_data, desc="clean covid data"):
            new_references = []
            for ref_id in paper["references"]:
                if ref_id in paper_ids:
                    new_references.append(ref_id)
            paper["references"] = list(set(new_references))
            paper["ref_list"] = paper["ref_list"]
            if len(new_references) > self.minimal_ref_num:
                new_covid_data.append(paper)
        print(f"original: {len(covid_data)}; now: {len(new_covid_data)}.")
        return new_covid_data

    def make_transductive_dataset(self):
        if not self.covid_data_path.exists():
            self.make_covid_data()
        covid_data = load_json(self.covid_data_path)
        cleaned_covid_data = self.clean_covid_data(covid_data)
        paper_list_set = [one["paper_id"] for one in cleaned_covid_data]
        save_json(paper_list_set, self.output_dir.joinpath('paper_list_set.json'))
        train_json, test_json, val_json = self._split_data(cleaned_covid_data, data_limit=100000)
        save_jsonl(train_json, self.output_dir.joinpath('train.jsonl'))
        save_jsonl(test_json, self.output_dir.joinpath('test.jsonl'))
        save_jsonl(val_json, self.output_dir.joinpath('val.jsonl'))
        print(f"make the transductive dataset to {self.output_dir}, "
              f"train: {len(train_json)}; val: {len(val_json)}; test: {len(test_json)}")


    def read_raw_covid_corpus(self, raw_data_dir: Path):
        samples = []
        data_path = raw_data_dir.joinpath('metadata.csv')
        with data_path.open("r", newline='', encoding="utf-8") as fr:
            reader = csv.reader(fr)
            next(reader)
            for line in tqdm(reader, desc=f"read_covid from {data_path}"):
                cord_uid = line[0]
                sha = line[1]
                source_x = line[2]
                title_ = line[3]
                doi = line[4]
                pmcid = line[5]
                pubmed_id = line[6]
                license = line[7]
                abstract = line[8]
                publish_time = line[9]
                authors = line[10]
                journal = line[11]
                mag_id = line[12]
                who_covidence_id = line[13]
                arxiv_id = line[14]
                pdf_json_files = [str(raw_data_dir.joinpath(file_name).absolute()) for file_name in
                                  line[15].split("; ")]
                pmc_json_files = [str(raw_data_dir.joinpath(file_name).absolute()) for file_name in
                                  line[16].split("; ")]
                url = line[17]
                s2_id = line[18]

                samples.append(
                    RawDataObject(cord_uid, sha, source_x, title_, doi, pmcid, pubmed_id, license, abstract,
                                  publish_time,
                                  authors, journal, mag_id, who_covidence_id, arxiv_id,
                                  pdf_json_files, pmc_json_files, url,
                                  s2_id)
                )
            print(f"data are loaded from {raw_data_dir}, and the datasize is {len(samples)}")
            return samples

def make_paper_dict(data_dir: Path, force_flag=True):
    dict_path = data_dir.joinpath("paper_dict.json")
    if not force_flag and dict_path.exists():
        return

    covid_data_list = json.load(data_dir.joinpath("covid_data.json").open("r", encoding="utf-8"))
    covid_dict = {}
    for one in covid_data_list:
        covid_dict[one["paper_id"]] = one
    paper_list_set = load_json(data_dir.joinpath("paper_list_set.json"))
    paper_dict = {}
    for file_name in os.listdir(data_dir):
        if file_name not in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            continue
        with jsonlines.open(data_dir.joinpath(file_name)) as fr:
            for json_dict in tqdm(list(fr), desc=f"make_paper_dict from {file_name}"):
                paper_dict[json_dict['paper_id']] = json_dict
                # reference
                for ref_id in json_dict['references']:
                    target_obj = covid_dict[ref_id]
                    cleaned_refs = []
                    for ref_id_ in target_obj['references']:
                        if ref_id_ in paper_list_set:
                            cleaned_refs.append(ref_id_)
                    target_obj["references"] = cleaned_refs
                    paper_dict[ref_id] = target_obj
    save_json(paper_dict, dict_path)
    print(f"create paper dict; size: {len(paper_dict)} ")

# 33332359
if __name__ == '__main__':
    processor = CovidDataProcessor(raw_data_dir=Path(f"{BASE_DIR}/resources/raw/covid-2022-06-02"),
               output_dir=Path(f"{BASE_DIR}/datasets/summarisation/biomed_ref_dataset"))
    processor.make_transductive_dataset()
    make_paper_dict(data_dir=Path(f"{BASE_DIR}/datasets/summarisation/biomed_ref_dataset"))
