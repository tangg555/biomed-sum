"""
@Desc:
@Reference:
"""

import os
import csv
import json
from tqdm import tqdm
from pathlib import Path
import sys
import glob
import os
from pathlib import Path
import json
import numpy as np
from src.utils import nlg_eval_utils

class Lead3(object):
    def __int__(self):
        pass

    def text2sentence(self, text):
        return


    def run(self, data_dir: Path):
        test_data_file = data_dir / f"test.jsonl"
        data_list = test_data_file.open("r", encoding="utf-8").readlines()
        preds = []
        refs = []
        results = {}
        for line in tqdm(data_list, desc="lead3 running..."):
            one_obj = json.loads(line.strip())
            intro_text = one_obj["introduction"]
            intro_sents = intro_text.split('. ')
            lead3_text = '. '.join(intro_sents[:3])
            ref_summary = one_obj["abstract"]

            preds.append(lead3_text)
            refs.append(ref_summary)
        rouge_metrics = nlg_eval_utils.compute_metrics_for_summary(pred_lines=preds, tgt_lines=refs)
        results.update(rouge_metrics)
        print(f"========== lead3 results:\n {results}")
        return results