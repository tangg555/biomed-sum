# Improving Biomedical Abstractive Summarisation with Knowledge Aggregation from Citation Papers

This repository is the code and resources for the paper [Improving Biomedical Abstractive Summarisation with Knowledge Aggregation from Citation Papers](@article{tang2023improving,
  title={Improving Biomedical Abstractive Summarisation with Knowledge Aggregation from Citation Papers},
  author={Tang, Chen and Wang, Shun and Goldsack, Tomas and Lin, Chenghua},
  journal={arXiv preprint arXiv:2310.15684},
  year={2023}
}). 
It has been accepted by EMNLP 2023.

## Instructions

This project is mainly implemented with following tools:
- **Pytorch**
- [pytorch-lightning](https://www.pytorchlightning.ai/) framework
- The initial checkpoints of pretrained models come from [Hugginface](https://huggingface.co).

So if you want to run this code, you must have the following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/): Mine is 1.13.0. At the time of coding, Pytorch 2.0 was not available. In case you would like to use Pytorch, you need to address some issues of incompatibility.
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) (4.27.4)
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) (mine is 1.9.5)
- Other dependencies: running `pip install -r requirement.txt` 
- 
## Datasets 
### Directly Download Dataset and Resources
To reproduce our work you need to download following files:

- Download the processed data and unzip them to be `datasets/summarisation` directory: [dropbox](https://www.dropbox.com/scl/fi/gu6y1tlh6anix4w4sq704/biomed_ref_dataset.zip?rlkey=72rlzmuvjy6rnad3j9wl1hnq7&dl=0)
- We also provide the generation result of ChatGPT (uploaded to [dropbox](https://www.dropbox.com/scl/fi/zlgkv56t1ue9aj8on68c5/ChatGPT_gen.txt?rlkey=3ukwuq5mrva5yne8fs98ebonl&dl=0)) for your reference. ChatGPT is a powerful text generation software, and it provides a paid API.

### Preprocess Dataset From Scratch

- The raw data come from the work of Allenai [cord19](https://github.com/allenai/cord19). We select the final release of the covid-19 data (2022-06-02) as our raw data. 
- Put the raw data to the `resources/raw` directory, and it will be `resources/raw/covid-2022-06-02`.
- Run code to preprocess the data, where the code is located at `tasks/summarisation/raw_data_preprocess`.

### The introduction of the dataset
The structure of `datasets`should be like this:
```markdown
├── datasets/summarisation/biomed_ref_dataset
      └── `covid_data.json` intermediate file
      └── `paper_dict.txt` recording the key information of the citation networks 
      └── `paper_list_set.txt` intermediate file
      └── `test.jsonl` the test set
      └── `train.jsonl` the train set
      └── `val.jsonl` the validation set
```

## Quick Start

### 1. Install packages
you have to install **Pytorch** from their [homepage](https://pytorch.org/get-started/locally/) 
as it has different dependencies for CPU and GPU (GPU also has different CUDA versions).

```shell
pip install -r requirements.txt
```

### 2. Collect Datasets and Resources

As mentioned above.

### 3. Run the code for training or testing

Please refer to the command examples listed in `python_commands.sh`.

For example, to train our model:
```shell
python tasks/summarisation/train.py --model_name sumref_bart --experiment_name=sumref_pubmedbart-biomed\
  --learning_rate=1e-4 --train_batch_size=2 --eval_batch_size=8 --model_name_or_path=mse30/bart-base-finetuned-pubmed \
  --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=4  --num_sanity_val_steps=1 \
  --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset
```
to test our model:
```shell
python tasks/summarisation/test.py\
  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumref_pubmedbart-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name sumref_bart --experiment_name=sumref_pubmedbart-biomed --eval_beams 4
```

## Notation
In case you would like to set up the environment without changing local dependencies. 
I suggest either creating a virtual python environment by [anaconda](https://www.anaconda.com/download) or pulling an related image with [docker](https://www.docker.com/products/docker-desktop/).


## Citation
If you found this repository or paper is helpful to you, please cite our paper. 
Currently, we only have the arxiv citation:

This is the arxiv citation:
```angular2
@article{tang2023improving,
  title={Improving Biomedical Abstractive Summarisation with Knowledge Aggregation from Citation Papers},
  author={Tang, Chen and Wang, Shun and Goldsack, Tomas and Lin, Chenghua},
  journal={arXiv preprint arXiv:2310.15684},
  year={2023}
}
```

The "pdf" column in datasets are recorded with absolute paths. Thought this will not affect the experiments with our model, I suggest to run the ``preprocess`` code from scracth.

