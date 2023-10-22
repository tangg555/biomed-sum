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

from src.configuration.constants import BASE_DIR
import json
from src.utils.file_utils import load_json
from src.configuration.constants import BASE_DIR

def limit_words(content:str, max_len):
    words = content.strip().split()
    return " ".join(words[:max_len])

def generate_doc(paper: dict, doc_max_len: int):
    sections = paper["sections"]
    # concatenate section contents
    doc = " ".join([one[1] for one in sections])
    doc = limit_words(doc, doc_max_len)
    return doc

def _check(files: list):
    for file_path in files:
        if (not Path(file_path).exists()) or (not Path(file_path).is_file()):
            continue
        pdf = load_json(file_path)
        if pdf:
            for section in pdf['body_text']:
                if section['section'].lower() == 'introduction':
                    return True
    return False

def extract_info(pdf_json_files: list, idf_pdf_dic) -> dict:
    for pdf_path in pdf_json_files:
        if (not Path(pdf_path).exists()) or (not Path(pdf_path).is_file()):
            return None
        pdf = load_json(pdf_path)
        paper = {}
        body_text = pdf['body_text']
        bibs = pdf['bib_entries']
        references = []
        if 'introduction' not in [section['section'].lower() for section in body_text]:
            return None
        for section in body_text:
            if 'introduction' == section['section'].lower():
                intro = section['text']
                paper['introduction'] = intro
            for cite in section['cite_spans']:
                ref_id = cite['ref_id']
                if ref_id not in bibs.keys():
                    continue
                if bibs[ref_id]['title'] in idf_pdf_dic.keys():
                    ref_files = idf_pdf_dic[bibs[ref_id]['title']]['files']
                    if _check(ref_files):
                        ref_id = idf_pdf_dic[bibs[ref_id]['title']]['id']
                        if len(ref_id) > 0:
                            references.append(ref_id)
        if len(references) == 0:
            continue
        paper['references'] = references
        return paper
    return None


def extract_doc_content(pdf_json_files: list, idf_pdf_dic):
    for pdf_path in pdf_json_files:
        if (not Path(pdf_path).exists()) or (not Path(pdf_path).is_file()):
            return None
        pdf = load_json(pdf_path)
        paper = {}
        body_text = pdf['body_text']
        bibs = pdf['bib_entries']
        references = []
        if 'introduction' not in [section['section'].lower() for section in body_text]:
            return None
        all_sections = [(section['section'].lower(), section['text'].lower()) for section in body_text]
        cleaned_sections = []
        for (sec_name, sec_content) in all_sections:
            if sec_name == "":
                continue
            cleaned_sections.append((sec_name, sec_content))
        for section in body_text:
            if 'introduction' == section['section'].lower():
                intro = section['text']
                paper['introduction'] = intro
            for cite in section['cite_spans']:
                ref_id = cite['ref_id']
                if ref_id not in bibs.keys():
                    continue
                if bibs[ref_id]['title'] in idf_pdf_dic.keys():
                    ref_files = idf_pdf_dic[bibs[ref_id]['title']]['files']
                    if _check(ref_files):
                        ref_id = idf_pdf_dic[bibs[ref_id]['title']]['id']
                        if len(ref_id) > 0:
                            references.append(ref_id)
        if len(references) == 0:
            continue
        paper['section_nums'] = len(cleaned_sections)
        paper['sections'] = cleaned_sections
        paper['references'] = references
        return paper
    return None