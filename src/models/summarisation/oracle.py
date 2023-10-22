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
from rouge import Rouge
from pathlib import Path
import json
import numpy as np
from src.utils import nlg_eval_utils
import copy
import concurrent.futures

from src.utils.summarisation.data_utils import generate_doc

class Oracle(object):
    max_workers = 64

    def simple_rouge(self, hyps, refs):
        rouge = Rouge()
        hyps = hyps
        refs = refs
        try:
            rouge_score = rouge.get_scores(hyps=hyps, refs=refs)
        except ValueError:
            return None
        return rouge_score

    def simple_rouge_eval(self, hyps, refer):
        r = self.simple_rouge(hyps, refer)
        if r == None:
            return 0
        return (r[0]['rouge-1']['f'] + r[0]['rouge-2']['f'] + r[0]['rouge-l']['f']) / 3

    def calLabel(self, article, abstract):
        hyps_list = article
        refer = abstract
        scores = []
        for hyps in hyps_list:
            if len(hyps) == 0:
                scores.append(0)
                continue
            mean_score = self.simple_rouge_eval(hyps, refer)
            scores.append(mean_score)

        selected = [int(np.argmax(scores))]
        selected_sent_cnt = 1

        best_rouge = np.max(scores)
        while selected_sent_cnt < len(hyps_list):
            cur_max_rouge = 0.0
            cur_max_idx = -1
            for i in range(len(hyps_list)):
                if i not in selected:
                    temp = copy.deepcopy(selected)
                    temp.append(i)
                    hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                    cur_rouge = self.simple_rouge_eval(hyps, refer)
                    if cur_rouge > cur_max_rouge:
                        cur_max_rouge = cur_rouge
                        cur_max_idx = i
            if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
                selected.append(cur_max_idx)
                selected_sent_cnt += 1
                best_rouge = cur_max_rouge
            else:
                break
        # print(selected, best_rouge)
        return selected

    def make_oracle_pair(self, line:str, data_idx=None, total_data_size=None):
        one_obj = json.loads(line.strip())
        input_text = generate_doc(one_obj, doc_max_len=4096)
        ref_summary = one_obj["abstract"]

        doc_sents = input_text.split('. ')
        refs = ref_summary
        r_scores = self.simple_rouge('. '.join(doc_sents), refs)
        if r_scores == None:
            return None
        ora = [doc_sents[i] for i in self.calLabel(doc_sents, refs)]
        ora_text = '. '.join(ora)
        if data_idx is not None:
            print(f"current progress: data_idx: {data_idx}; total_data_size:{total_data_size}")
        return ora_text, ref_summary


    def run(self, data_dir: Path):
        test_data_file = data_dir / f"test.jsonl"
        data_list = test_data_file.open("r", encoding="utf-8").readlines()
        results = {}
        # ====== run with multiprocessing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 使用map方法，将输入数据分配给不同的线程，并获取返回值
            print(f"使用executor开启多线程; max_workers:{self.max_workers}")
            thread_results = executor.map(self.make_oracle_pair,
                                          data_list,
                                          [i for i in range(len(data_list))],
                                          [len(data_list)] * len(data_list),
                                          )
            thread_results: list = list(thread_results)
            if None in thread_results:
                thread_results.remove(None)
            preds = [one[0] for one in thread_results]
            refs = [one[1] for one in thread_results]
        rouge_metrics = nlg_eval_utils.compute_metrics_for_summary(pred_lines=preds, tgt_lines=refs)
        results.update(rouge_metrics)
        print(f"========== oracle results:\n {results}")
        return results