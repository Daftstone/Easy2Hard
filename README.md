<<<<<<< HEAD
# Easy to Hard Enhancement

This is the source code for the paper "Advancing Machine-Generated Text Detection from an Easy to Hard Supervision Perspective", which is built on [[MGTBench]](https://github.com/xinleihe/MGTBench).

## Installation
```
conda env create -f environment.yml;
conda activate Detection;
mkdir save_models;
download gpt2-medium model from https://huggingface.co/openai-community/gpt2-medium, and move it to save_models/;
download chatgpt-detector-roberta model from https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta, and move it to save_models/;
download roberta-base-openai-detector model from https://huggingface.co/openai-community/roberta-base-openai-detector, and move it to save_models/;
download AIGC_detector_env1 model from https://huggingface.co/yuchuantian/AIGC_detector_env2, and move it to save_models/;
download gpt2-xl model from https://huggingface.co/yuchuantian/AIGC_detector_env2, and move it to save_models/;
download t5-base model from https://huggingface.co/google-t5/t5-base, and move it to save_models/;
```

## Train
```
usage: python benchmark.py --finetune [--method METHOD] [--dataset DATA_NAME] [--detectLLM TEST_TEXT]
               [--reg REG] [--sentence_num SENTENCE_NUM]
               [--sentence_length SENTENCE_LENGTH] [--iter ITER]

optional arguments:
  --method METHOD
                        Supported: Log-Likelihood, Log-Rank, NPR, DetectGPT, FastGPT, ChatGPT-D, MPU, RADAR
  --dataset DATA_NAME
                        Supported: Essay, DetectRL, Mix (mixed text scenario), polish and back_translation (paraphrasing attack scenario)
  --detectLLM TEST_TEXT
                        Supported: 
                          Essay: ChatGPT-turbo, ChatGLM, Dolly, ChatGPT, GPT4All, Claude;
                          DetectRL: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
                          Mix: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
                          polish: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
                          back_translation: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
  --sentence_num SENTENCE_NUM
                        text number k in longer text;
  --sentence_length SENTENCE_LENGTH
                        1 for sentence-level detection and -1 for paragraph-level detection;
  --reg REG
                        supervisor loss coefficient \lambda
  --iter ITER
                        random seed

Example: python benchmark.py --dataset Essay --detectLLM GPT4All --method RADAR --DEVICE cuda:0 --load_path Essay-ChatGPT --iter 0 --reg 10. --sentence_num 3 --sentence_length 1
```

## Inference
```
usage: python benchmark.py [--method METHOD] [--dataset DATA_NAME] [--detectLLM TEST_TEXT]
               [--reg REG] [--sentence_num SENTENCE_NUM]
               [--sentence_length SENTENCE_LENGTH] [--iter ITER] [--load_path LOAD_PATH]

optional arguments:
  --method METHOD
                        Supported: Log-Likelihood, Log-Rank, NPR, DetectGPT, FastGPT, ChatGPT-D, MPU, RADAR
  --dataset DATA_NAME
                        Supported: Essay, DetectRL, Mix (mixed text scenario), polish and back_translation (paraphrasing attack scenario)
  --detectLLM TEST_TEXT
                        Supported: 
                          Essay: ChatGPT-turbo, ChatGLM, Dolly, ChatGPT, GPT4All, Claude;
                          DetectRL: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
                          Mix: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
                          polish: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
                          back_translation: ChatGPT, Claude-instant, Google-PaLM, Llama-2-70b;
  --sentence_num SENTENCE_NUM
                        text number k in longer text;
  --sentence_length SENTENCE_LENGTH
                        1 for sentence-level detection and -1 for paragraph-level detection;
  --reg REG
                        supervisor loss coefficient \lambda
  --iter ITER
                        random seed
  --load_path LOAD_PATH
                        It is the path where the training model is laoded, and its format is Dataset-LLM, e.g., Essay-GPT4All

Example: python benchmark.py --dataset Essay --detectLLM GPT4All --method RADAR --DEVICE cuda:0 --load_path Essay-ChatGPT --iter 0 --reg 10. --sentence_num 3 --sentence_length 1
```
=======
# MGTBench

MGTBench provides the reference implementations of different machine-generated text (MGT) detection methods.
It is still under continuous development and we will include more detection methods as well as analysis tools in the future.


## Supported Methods
Currently, we support the following methods (continuous updating):
- Metric-based methods:
    - Log-Likelihood [[Ref]](https://arxiv.org/abs/1908.09203);
    - Rank [[Ref]](https://arxiv.org/abs/1906.04043);
    - Log-Rank [[Ref]](https://arxiv.org/abs/2301.11305);
    - Entropy [[Ref]](https://arxiv.org/abs/1906.04043);
    - GLTR Test 2 Features (Rank Counting) [[Ref]](https://arxiv.org/abs/1906.04043);
    - DetectGPT [[Ref]](https://arxiv.org/abs/2301.11305);
    - LRR [[Ref]](https://arxiv.org/abs/2306.05540);
    - NPR [[Ref]](https://arxiv.org/abs/2306.05540);
- Model-based methods:
    - OpenAI Detector [[Ref]](https://arxiv.org/abs/1908.09203);
    - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
    - ConDA [[Ref]](https://arxiv.org/abs/2309.03992) [[Model Weights]](https://www.dropbox.com/s/sgwiucl1x7p7xsx/fair_wmt19_chatgpt_syn_rep_loss1.pt?dl=0);
    - GPTZero [[Ref]](https://gptzero.me/);
    - LM Detector [[Ref]](https://arxiv.org/abs/1911.00650);

## Supported Datasets
- Essay;
- WP;
- Reuters; 

Note that our datasets are constructed based on [Verma et al.](https://arxiv.org/abs/2305.15047), you can download them from [Google Drive](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola?usp=sharing).

## Installation
```
git clone https://github.com/xinleihe/MGTBench.git;
cd MGTBench;
conda env create -f environment.yml;
conda activate MGTBench;
```

## Usage
To run the benchmark on the Essay dataset: 
```
# Distinguish Human vs. Claude:
python benchmark.py --dataset Essay --detectLLM Claude --method Log-Likelihood

# Text attribution:
python attribution_benchmark.py --dataset Essay
```
Note that you can also specify your own datasets on ``dataset_loader.py``.


## Authors
The tool is designed and developed by Xinlei He (CISPA), Xinyue Shen (CISPA), Zeyuan Chen (Individual Researcher), Michael Backes (CISPA), and Yang Zhang (CISPA).

## Cite
If you use MGTBench for your research, please cite [MGTBench: Benchmarking Machine-Generated Text Detection](https://arxiv.org/abs/2303.14822).

```
bibtex
@article{HSCBZ23,
author = {Xinlei He and Xinyue Shen and Zeyuan Chen and Michael Backes and Yang Zhang},
title = {{MGTBench: Benchmarking Machine-Generated Text Detection}},
journal = {{CoRR abs/2303.14822}},
year = {2023}
}
```
>>>>>>> 4e120c240e58148c1773c572967df73ee2d8f827
