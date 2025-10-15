# Easy to Hard MGT Detection Enhancement

This is the source code for the paper "Advancing Machine-Generated Text Detection from an Easy to Hard Supervision Perspective" (NeurIPS 2025).

<img width="2343" height="681" alt="image" src="https://github.com/user-attachments/assets/8f62dae3-c735-431a-8eae-8c1b25b7a05d" />

## Introduction
To address the misuse of high-quality Machine Generated Text (MGT) in areas such as misinformation dissemination and phishing, developing powerful MGT detection methods has become critically important. However, current methods typically rely on an assumption of “gold standard” data labels, which may not hold in today’s MGT detection landscape. Specifically, the widespread adoption of human–machine collaboration and sharp advances in Large Language Models (LLMs) have significantly blurred the boundary between human and machine-generated text, leading to inherently imprecise labels. More challenging still, humans exhibit only limited accuracy in identifying MGT (barely surpassing random guessing), while existing detectors have already demonstrated “super-intelligence,” surpassing human performance. This creates substantial difficulties in both providing and evaluating higher-quality supervision signals.

This situation highlights a fundamental challenge in MGT detection: the common and inevitable challenge of “inexact supervision.” To address this, we propose a novel training paradigm aimed at providing an efficient path toward building more powerful MGT detectors.

Our new paradigm employs an “easy-to-hard” supervision perspective. The core idea is to leverage an "easy supervisor", designed for a more easy detection task but with greater reliability, to provide robust supervision signals to a "target detector" that handles more complex detection tasks. This approach overturns the traditional “teacher–student” knowledge distillation concept, where a stronger teacher instructs a weaker student, by harnessing a model that excels at a simpler task to bolster a more comprehensive detector. 

Experimental results indicate that this framework significantly boosts the performance of existing detectors across various complex scenarios, such as cross-LLM, cross-domain, mixed text, and paraphrase attacks, while introducing nearly no extra training latency.

## Supported Methods
The project supports the following methods:
- Metric-based methods:
    - Log-Likelihood [[Ref]](https://arxiv.org/abs/1908.09203);
    - Log-Rank [[Ref]](https://arxiv.org/abs/2301.11305);
    - DetectGPT [[Ref]](https://arxiv.org/abs/2301.11305);
    - NPR [[Ref]](https://arxiv.org/abs/2306.05540);
    - FastGPT [[Ref]](https://arxiv.org/abs/2310.05130);
- Model-based methods:
    - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
    - MPU [[Ref]](https://arxiv.org/abs/2305.18149);
    - RADAR [[Ref]](https://proceedings.neurips.cc/paper_files/paper/2023/file/30e15e5941ae0cdab7ef58cc8d59a4ca-Paper-Conference.pdf);

## Supported Datasets
- Essay [[Ref]](https://arxiv.org/abs/2305.15047);
- DetectRL [[Ref]](https://proceedings.neurips.cc/paper_files/paper/2024/file/b61bdf7e9f64c04ec75a26e781e2ad51-Paper-Datasets_and_Benchmarks_Track.pdf);

## Environmental Installation
```
conda env create -f environment.yml;
conda activate Detection;
```

## Model Download
Download the necessary models to the save_models folder.
```
download gpt2-medium model from https://huggingface.co/openai-community/gpt2-medium, and move it to save_models/;
download chatgpt-detector-roberta model from https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta, and move it to save_models/;
download roberta-base-openai-detector model from https://huggingface.co/openai-community/roberta-base-openai-detector, and move it to save_models/;
download AIGC_detector_env1 model from https://huggingface.co/yuchuantian/AIGC_detector_env2, and move it to save_models/;
download gpt2-xl model from https://huggingface.co/openai-community/gpt2-xl, and move it to save_models/;
download t5-base model from https://huggingface.co/google-t5/t5-base, and move it to save_models/;
```

## Detector Training
```
usage: python benchmark.py --finetune [--method METHOD] [--dataset DATA_NAME] [--detectLLM TEST_TEXT]
               [--reg REG] [--sentence_num SENTENCE_NUM]
               [--sentence_length SENTENCE_LENGTH] [--iter ITER]

arguments:
  --method METHOD
                        Supported: Log-Likelihood, Log-Rank, NPR, DetectGPT, FastGPT, ChatGPT-D, MPU, RADAR
  --dataset DATA_NAME
                        Supported: Essay, DetectRL, Mix (mixed text scenario), Polish and Back_translation (paraphrasing attack scenario)
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

Example: python benchmark.py --dataset Essay --detectLLM GPT4All --method RADAR --DEVICE cuda:0 --finetune --iter 0 --reg 10. --sentence_num 3 --sentence_length 1
```

## Detector Inference
```
usage: python benchmark.py [--method METHOD] [--dataset DATA_NAME] [--detectLLM TEST_TEXT]
               [--reg REG] [--sentence_num SENTENCE_NUM]
               [--sentence_length SENTENCE_LENGTH] [--iter ITER] [--load_path LOAD_PATH]

arguments:
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
                        It is the path where the training model is loaded, and its format is Dataset-LLM, e.g., Essay-GPT4All

Example: python benchmark.py --dataset Essay --detectLLM GPT4All --method RADAR --DEVICE cuda:0 --load_path Essay-ChatGPT --iter 0 --reg 10. --sentence_num 3 --sentence_length 1
```

## Acknowledgements
This project is built upon [[MGTBench]](https://github.com/xinleihe/MGTBench).
