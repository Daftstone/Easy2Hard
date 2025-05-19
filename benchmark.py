from parse import args
import datetime
import os
import json
import dataset_loader_sentence
from methods.utils import load_base_model, load_base_model_and_tokenizer
from methods.supervised import run_supervised_experiment_New
from methods.detectgpt import run_perturbation_experiments
from methods.supervised import run_fast_experiments
from methods.gptzero import run_gptzero_experiment
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, \
    run_GLTR_experiment

if __name__ == '__main__':

    if (args.sentence_length == -1):
        args.batch_size = 16
        args.paragraph_num = 32

    import random
    import torch

    random.seed(0)
    seeds = [random.randint(0, 100000000) for _ in range(100)]

    random.seed(seeds[args.iter])
    torch.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed_all(seeds[args.iter])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading dataset {args.dataset}...')
    data = dataset_loader_sentence.load(args.dataset, detectLLM=args.detectLLM)

    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.dataset}-{args.detectLLM}"
    if (len(args.load_path) > 0):
        temp = args.load_path.split("-")
        if (len(temp) == 2):
            train_model = temp[1]
        else:
            train_model = "-".join(temp[1:])
    else:
        train_model = args.detectLLM
    args.train_model = train_model
    if (len(args.load_path) > 0):
        args.load_path_criterion = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{train_model}_{args.method}_benchmark_results_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{args.reg}_{args.iter}.pkl"
    if (len(args.load_path) > 0):
        args.load_path = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{args.method}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(SAVE_PATH + '/%s' % args.method):
        os.makedirs(SAVE_PATH + '/%s' % args.method)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    if (args.method in ['Log-Likelihood', 'Log-Rank', 'Entropy', 'FastGPT', 'DetectGPT', 'NPR']):
        base_model, base_tokenizer = load_base_model_and_tokenizer(
            args.base_model_name, cache_dir)
        load_base_model(base_model, DEVICE)


    def ll_criterion(text):
        return get_ll(
            text, base_model, base_tokenizer, DEVICE)


    def rank_criterion(text):
        return -get_rank(text,
                         base_model, base_tokenizer, DEVICE, log=False)


    def logrank_criterion(text):
        return -get_rank(text,
                         base_model, base_tokenizer, DEVICE, log=True)


    def entropy_criterion(text):
        return get_entropy(
            text, base_model, base_tokenizer, DEVICE)


    def GLTR_criterion(text):
        return get_rank_GLTR(
            text, base_model, base_tokenizer, DEVICE)


    outputs = []
    # outputs.append(run_threshold_experiment(data, phd_criterion, "phd"))

    if args.method == "Log-Likelihood":
        outputs.append(run_threshold_experiment(
            data, ll_criterion, "likelihood", args.load_path_criterion))
    elif args.method == "Rank":
        outputs.append(run_threshold_experiment(data, rank_criterion, "rank", args.load_path_criterion))
    elif args.method == "Log-Rank":
        outputs.append(run_threshold_experiment(
            data, logrank_criterion, "log_rank", args.load_path_criterion))
    elif args.method == "Entropy":
        outputs.append(run_threshold_experiment(
            data, entropy_criterion, "entropy", args.load_path_criterion))
    elif args.method == "GLTR":
        outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", args.load_path_criterion))
    elif args.method == "FastGPT":
        score_model, score_tokenizer = load_base_model_and_tokenizer(
            'gpt2-xl', cache_dir)
        load_base_model(score_model, DEVICE)
        outputs.append(run_fast_experiments(
            data, base_model, base_tokenizer, score_model, score_tokenizer))
    elif args.method == "DetectGPT":
        outputs.append(run_perturbation_experiments(
            args, data, base_model, base_tokenizer, args.load_path_criterion, method="DetectGPT"))
    elif args.method == "NPR":
        outputs.append(run_perturbation_experiments(
            args, data, base_model, base_tokenizer, args.load_path_criterion, method="NPR"))


    elif args.method == "RADAR":
        outputs.append(
            run_supervised_experiment_New(
                data,
                model='RADAR',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=0,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/RADAR",
                load_path=args.load_path))
    elif args.method == "ChatGPT_D":
        outputs.append(
            run_supervised_experiment_New(
                data,
                model='ChatGPT_D',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/ChatGPT_D",
                load_path=args.load_path))
    elif args.method == "MPU":
        outputs.append(
            run_supervised_experiment_New(
                data,
                model='MPU',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=args.finetune,
                save_path=SAVE_PATH +
                          f"/MPU",
                load_path=args.load_path))

    # # run GPTZero: pleaze specify your gptzero_key in the args
    elif args.method == "GPTZero":
        outputs.append(run_gptzero_experiment(data, api_key=args.gptzero_key))

    # save results
    import pickle as pkl

    with open(os.path.join(SAVE_PATH,
                           f"{train_model}_{args.method}_benchmark_results_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{args.reg}_{args.iter}.pkl"),
              "wb") as f:
        pkl.dump(outputs, f)

    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    print("Finish")
