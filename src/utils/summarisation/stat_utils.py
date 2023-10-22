"""
@Desc:
@Reference:
@Notes:
"""

import os
import sys
from pathlib import Path

from collections import Counter
from typing import List

import json
from src.utils.file_utils import load_json


def round_dict(dict_: dict):
    for key in dict_:
        dict_[key] = round(dict_[key], 2)
    return dict_

def basic_info(data_list: list):
    basic_info_dict = {"data_size": 0,
                       "paper_num": 0}
    paper_list = []
    for line in data_list:
        one_obj = json.loads(line.strip())
        paper_list.extend(one_obj["references"])
        paper_list.append(one_obj["paper_id"])
        basic_info_dict["data_size"] += 1
    basic_info_dict["paper_num"] = len(set(paper_list))
    return round_dict(basic_info_dict)

def detail_info(data_list: list, data_size: int):
    detail_stat_dict = {
        "total_citations": 0,
        "distinct_citations": 0,
        "input_sections": 0,

        "doc_sents": 0,
        "doc_words": 0,
        "doc_sents": 0,
        "doc_words": 0,

        "summary_sents": 0,
        "summary_words": 0
    }
    for line in data_list:
        one_obj = json.loads(line.strip())
        doc_text = "\n".join([one[1] for one in one_obj["sections"]])
        summary_text = one_obj["abstract"]
        
        detail_stat_dict["input_sections"] += one_obj["section_nums"]
        detail_stat_dict["distinct_citations"] += len(one_obj["references"])
        detail_stat_dict["total_citations"] += len(one_obj["ref_list"])
        
        doc_sents = doc_text.strip().split(".")
        doc_words = doc_text.strip().split()
        detail_stat_dict["doc_sents"] += len(doc_sents)
        detail_stat_dict["doc_words"] += len(doc_words)

        doc_sents = doc_text.strip().split(".")
        doc_words = doc_text.strip().split()
        detail_stat_dict["doc_sents"] += len(doc_sents)
        detail_stat_dict["doc_words"] += len(doc_words)

        summary_sents = summary_text.strip().split(".")
        summary_words = summary_text.strip().split()
        detail_stat_dict["summary_sents"] += len(summary_sents)
        detail_stat_dict["summary_words"] += len(summary_words)
    for key in detail_stat_dict:
        detail_stat_dict[key] /= data_size
    return round_dict(detail_stat_dict)

if __name__ == '__main__':
    basic_info([])
