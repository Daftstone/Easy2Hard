# Easy to Hard MGT Detection Enhancement

This is the source code for the paper "Advancing Machine-Generated Text Detection from an Easy to Hard Supervision Perspective" (NeurIPS 2025), which is built on [[MGTBench]](https://github.com/xinleihe/MGTBench).

<img width="2343" height="681" alt="image" src="https://github.com/user-attachments/assets/8f62dae3-c735-431a-8eae-8c1b25b7a05d" />

## Introduction
To counter the misuse of high-quality Machine-Generated Text (MGT) in areas like disinformation campaigns and phishing attacks, developing powerful MGT detection has become crucial. However, current detection methods generally presume the availability of "gold-standard" data labels—an assumption that is increasingly untenable in the current landscape. Specifically, the growing prevalence of human-AI collaborative writing and the leap in the generative capabilities of Large Language Models (LLMs) are making the boundary between human and machine text increasingly ambiguous. This ambiguity introduces inherent inexactness into the supervisory labels used for training. More challenging still is that human ability to recognize MGT is limited (performing only slightly better than chance), whereas existing detectors already exhibit superhuman capabilities, making the provision and assessment of higher-quality supervision a formidable obstacle.

This reality highlights a central challenge in the field of MGT detection: a widespread and inescapable problem of "Inexact Supervision." To confront this challenge, we propose a new training paradigm that opens up an efficient path toward building more powerful MGT detectors.

Our new paradigm is realized through an "Easy-to-Hard" supervision perspective. Its central tenet is that by designing a "Supervisor" model focused on a simpler task where it can achieve higher reliability, it can in turn provide more robust supervisory signals for a "target Detector" that handles more complex tasks. This method subverts the traditional "strong-teacher-for-weak-student" concept in knowledge distillation. Instead, it utilizes a model that is more reliable on a specific, simpler task to empower a target model with stronger overall capabilities. 

Experimental results demonstrate that this framework significantly boosts the performance of existing detectors across diverse and complex scenarios—including cross-LLM, cross-domain, hybrid human-AI texts, and under paraphrasing attacks—while introducing almost no additional training latency.

## Supported Methods
Currently, we support the following methods (continuous updating):
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
                        It is the path where the training model is laoded, and its format is Dataset-LLM, e.g., Essay-GPT4All

Example: python benchmark.py --dataset Essay --detectLLM GPT4All --method RADAR --DEVICE cuda:0 --load_path Essay-ChatGPT --iter 0 --reg 10. --sentence_num 3 --sentence_length 1
```
