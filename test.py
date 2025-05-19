import os
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--DEVICE', type=str, default="cuda:0")
parser.add_argument('--method', type=str, default="EM")
parser.add_argument('--dataset', type=str, default="Essay")
parser.add_argument('--train_model', type=str, default="ChatGPT-turbo")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--sentence_num', type=int, default=4)
parser.add_argument('--sentence_length', type=int, default=1)
parser.add_argument('--adversarial', type=int, default=1)

args = parser.parse_args()

if (args.dataset == 'TruthfulQA' or args.dataset == 'NarrativeQA' or args.dataset == 'SQuAD1'):
    LLMS = ['ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4', 'StableLM']
elif (args.dataset == 'RAID'):
    LLMS = ['mistral-chat', 'gpt3', 'gpt4', 'mpt', 'cohere-chat', 'gpt2', 'chatgpt', 'mistral', 'llama-chat',
            'cohere', 'mpt-chat']
    LLMS = ['gpt3', 'gpt4', 'mpt', 'gpt2', 'chatgpt', 'mistral', 'cohere']
elif (args.dataset in ['DetectRL', 'Mix', 'dipper', 'polish', 'CMix', 'back_translation', 'CCMix']):
    LLMS = ['ChatGPT', 'Claude-instant', 'Google-PaLM', 'Llama-2-70b']
elif (args.dataset in ['Cross']):
    LLMS = ['arxiv', 'writing_prompt', 'xsum', 'yelp_review']
else:
    LLMS = ['ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'Claude']

for i in range(args.epochs):
# for i in [int(args.DEVICE[-1:])]:
#     os.system(
#         "python benchmark.py --dataset %s --detectLLM %s --method %s --DEVICE %s --finetune --iter %d --reg %s --sentence_num %d --sentence_length %d --adversarial %d" % (
#             args.dataset, args.train_model, args.method, args.DEVICE, i, args.reg, args.sentence_num,
#             args.sentence_length, args.adversarial))
#
#     for LLM_model in LLMS:
#         os.system(
#             "python benchmark.py --dataset %s --detectLLM %s --method %s --DEVICE %s --load_path %s-%s --iter %d --reg %s --sentence_num %d --sentence_length %d --adversarial %d" % (
#                 args.dataset, LLM_model, args.method, args.DEVICE, args.dataset, args.train_model, i, args.reg,
#                 args.sentence_num, args.sentence_length, args.adversarial))

    for LLM_model in LLMS:
        with open(os.path.join(f"update_results/gpt2-medium-t5-base/{args.dataset}-{LLM_model}",
                               f"{args.train_model}_{args.method}_benchmark_results_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{args.reg}_{i}.pkl"),
                  "rb") as f:
            output = pkl.load(f)
            print(LLM_model, output[0]['general']['acc_test'], output[0]['general']['auc_test'],
                  output[0]['general']['tpr1_test'], output[0]['general']['tpr2_test'],
                  output[0]['general']['tpr3_test'])
    os.system(
        f'rm update_results/gpt2-medium-t5-base/{args.dataset}-{args.train_model}/{args.method}/model_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{args.reg}_{i}.pth')
