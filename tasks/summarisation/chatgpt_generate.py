"""
@Desc:
@Reference:
@Notes:
WANDB is Weights and Biases Logger:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html
"""

import sys
import json
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import openai
import concurrent.futures
import os

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from transformers import BartTokenizer

from src.utils.file_utils import copy_file_or_dir, output_obj_to_file, load_jsonl, load_json, save_lines
from src.utils import nlg_eval_utils
from src.utils.summarisation.data_utils import generate_doc, limit_words
from retrying import retry

class ChatgptGenerator(object):
    def __init__(self, dataset_dir:Path):
        # parameters

        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.generation_dir = None
        self.gen_file = None
        self.eval_file = None
        self.dataset_dir = None

        self.src_file = None
        self.tgt_file = None
        self.gen_file = None

        self.model_name = "gpt-3.5-turbo"

        openai.api_key = os.getenv("CHAT_API")
        self.max_workers = 32
        self.dataset_dir = dataset_dir

        self.max_doc_length = 766
        self.max_abs_length = 256

    def compose_prompts_no_ref(self, test_data: list):
        input_texts = []
        for paper in test_data:
            doc = generate_doc(paper, doc_max_len=self.max_doc_length)
            input_text = f"Write a summary according to given paper content, " \
                         f"which is part of a medical scientific paper (A). " \
                         f"The length of generated summary is expected to be larger than 130 words and less than {self.max_abs_length} words.\n" \
                         f"The paper content of (A) is: {doc}\n\n" \
                         f"Output:"
            input_text = json.dumps(input_text)
            input_texts.append(input_text)
        return input_texts

    def compose_prompts_one_ref(self, test_data: list, paper_dict: dict):
        input_texts = []
        for paper in test_data:
            doc = generate_doc(paper, doc_max_len=self.max_doc_length)
            ref_abs_list = paper["references"]
            paper_id = ref_abs_list[0]
            ref_abs = paper_dict[paper_id]["abstract"]
            input_text = f"Write a summary according to a given paper content (word limit is {self.max_doc_length}) " \
                         f"which is part of a medical scientific paper (A)," \
                         f"and an abstract as the reference from another scientific paper cited by this paper (A). " \
                         f"The length of generated summary is expected to be larger than 130 words and less than {self.max_abs_length} words.\n" \
                         f"The given paper content is:  {doc}\n\n" \
                         f"The abstract as the reference: {ref_abs} \n\n" \
                         f"Output:"
            input_text = json.dumps(input_text)
            input_texts.append(input_text)
        return input_texts

    def model_generate(self, mode="no_ref"):
        """
        generate stories according to given conditions
        """
        assert mode in ["no_ref", "one_ref", "multi_ref"]
        model_name = self.model_name
        self.generation_dir = Path(f"{BASE_DIR}/output/summarisation/{mode}_{model_name}_gen_result")
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.gen_file = self.generation_dir / f"{model_name}_gen.txt"
        self.eval_file = self.generation_dir / f"{model_name}_eval.txt"

        test_file_path = self.dataset_dir / "test.jsonl"
        copy_file_or_dir(test_file_path, self.generation_dir / "test.jsonl")

        test_data = load_jsonl(test_file_path)

        if mode == "no_ref":
            input_texts = self.compose_prompts_no_ref(test_data)
        elif mode == "one_ref":
            paper_dict_path = self.dataset_dir / "paper_dict.json"
            paper_dict = load_json(paper_dict_path)
            input_texts = self.compose_prompts_one_ref(test_data, paper_dict)
        else:
            raise NotImplementedError()


        ref_summaries = [one["abstract"] for one in test_data]
        prompt_file = self.generation_dir / "prompts.txt"
        save_lines(input_texts, prompt_file)
        generated_summaries = self.chatgpt_batch_func_with_cache(input_texts, self.gen_file, self.max_workers)
        metrics = self.eval_output(pred_lines=generated_summaries, tgt_lines=ref_summaries)
        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)

        print(f"==== eval output ====\n{json.dumps(metrics, indent=4)} ")


    def chatgpt_batch_func_with_cache(self, input_texts: list, tgt_file: Path, max_workers=512):
        results = []
        if tgt_file.exists():
            with tgt_file.open("r", encoding="utf-8") as fr:
                results = fr.readlines()
        len_results = len(results)
        print(f"{len_results} has already been generated.")
        with tgt_file.open("w", encoding="utf-8") as fw:
            fw.writelines(results)
            fw.flush()
            # for input_text in tqdm(input_texts[len_results:], desc="chatgpt_func_with_cache"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 使用map方法，将输入数据分配给不同的线程，并获取返回值
                print(f"使用executor开启多线程; max_workers:{max_workers}")
                data = input_texts[len_results:]
                thread_results = executor.map(self.chatgpt_func,
                                              data,
                                              [i for i in range(len(input_texts))],
                                              [len(input_texts)] * len(input_texts)
                                              )
                for response in thread_results:
                    response = response.replace("\n", "") + "\n"
                    fw.write(response)
                    results.append(response)
                    fw.flush()
        return results


    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def chatgpt_func(self, content, data_idx=None, total_data_size=None):
        message = {'role': 'user', 'content': content}
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            messages=[message],
            temperature=0.0,
            # max_tokens=max_tokens,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=stop_sequence,
            model=self.model_name,
        )
        response_text = response["choices"][0]["message"]["content"].strip()
        if data_idx is not None:
            print(f"current progress: data_idx: {data_idx}; total_data_size:{total_data_size}")
        return response_text


    def eval_output(self, pred_lines: list, tgt_lines: list):
        metrics = nlg_eval_utils.compute_metrics_for_summary(pred_lines=pred_lines, tgt_lines=tgt_lines)
        return metrics

if __name__ == '__main__':
    chatgpt = ChatgptGenerator(dataset_dir=Path(f"{BASE_DIR}/datasets/summarisation/biomed_ref_dataset"))
    chatgpt.max_workers = 6
    chatgpt.max_retry_times = 10
    # generate summaries
    # chatgpt.model_generate(mode="no_ref")
    chatgpt.model_generate(mode="one_ref")

