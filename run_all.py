import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Essay")
parser.add_argument('--method', type=str, default="EM")
parser.add_argument('--gpu', type=str, default="cuda:2")
parser.add_argument('--sentence_num', type=int, default=3)
parser.add_argument('--sentence_length', type=int, default=1)
parser.add_argument('--adversarial', type=int, default=1)
parser.add_argument('--reg', type=float, default=0.1)

args = parser.parse_args()

if (args.dataset == 'TruthfulQA' or args.dataset == 'NarrativeQA' or args.dataset == 'SQuAD1'):
    llms = ['ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4', 'StableLM', 'BloomZ']
    llms = ['ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4', 'StableLM']
    llms = ['ChatGPT-turbo', 'ChatGLM', 'Dolly']
elif (args.dataset == 'RAID'):
    llms = ['mistral-chat', 'gpt3', 'gpt4', 'mpt', 'cohere-chat', 'gpt2', 'chatgpt', 'mistral', 'llama-chat',
            'cohere', 'mpt-chat']
    llms = ['gpt3', 'gpt4', 'mpt', 'gpt2', 'chatgpt', 'mistral', 'cohere']
elif (args.dataset == 'DetectRL' or args.dataset == 'Mix' or args.dataset == 'CMix' or args.dataset == 'CCMix'):
    llms = ['ChatGPT', 'Claude-instant', 'Google-PaLM', 'Llama-2-70b']
    llms = ['Google-PaLM']
else:
    llms = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM']
    llms = ['ChatGPT-turbo', 'ChatGLM', 'Claude', 'Dolly', 'ChatGPT', 'GPT4All']
    llms = ['GPT4All']

for llm in llms:
    os.system(
        "python test.py --DEVICE %s --method %s --dataset %s --train_model %s --sentence_num %d --sentence_length %d --adversarial %d --reg %s| tee logs/temp_%s_%s_%s_%d_%d_%d_%s.txt" % (
            args.gpu, args.method, args.dataset, llm, args.sentence_num, args.sentence_length, args.adversarial,
            args.reg, args.dataset,
            llm, args.method, args.sentence_num, args.sentence_length, args.adversarial, args.reg))
