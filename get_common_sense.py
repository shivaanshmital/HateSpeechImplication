## To run: python scripts/interactive/getCommonSense.py --model_file pretrained_models/pretrained_models/atomic_pretrained_model.pickle

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()

    # opt, state_dict = interactive.load_model_file(args.model_file)

    # data_loader, text_encoder = interactive.load_data("atomic", opt)

    # n_ctx = data_loader.max_event + data_loader.max_effect
    # n_vocab = len(text_encoder.encoder) + n_ctx
    # model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    model_file = "./pretrained_models/atomic_pretrained_model.pickle"
    opt, state_dict = interactive.load_model_file(model_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    print("Using device --> ", cfg.device)
    
    data_path = "./Data_new/"
    with open(data_path + "trainj.json", "r") as f:
        train = json.load(f)
    with open(data_path + "valj.json", "r") as f:
        val = json.load(f)
    with open(data_path + "testj.json", "r") as f:
        test = json.load(f)
    
    sampler = interactive.set_sampler(opt, "topk-10", data_loader)
    category = "all"
    
    # Train
    for key,value in tqdm(train.items(), desc="Train"):
        try:
            topic = value["post"]
            if len(topic) > 50:
                topic = topic[0:50]
            outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
            print(outputs.keys())
            value["commonsense"] = outputs
        except Exception as e:
            print(e)
            print(key)
    with open(data_path + "train_with_commonsense_final.json","w") as f:
        json.dump(train,f)
        
    # # Val
    # for key,value in tqdm(val.items(), desc="Val"):
    #     try:
    #         topic = value["post"]
    #         if len(topic) > 50:
    #             topic = topic[0:50]
    #         outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         # with open("outputs.txt","w") as f:
    #         #     f.write(topic)
    #         #     f.write("\n----------------------")
    #         #     f.write(json.dumps(outputs))
    #         #     f.write("\n----------------------")
    #         # try:
    #         #     outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         # except Exception as e:
    #         #     topic = topic[0:min(50,len(topic))]
    #         #     outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         value["commonsense"] = outputs
    #     except Exception as e:
    #         print(e)
    #         print(key)
    # with open(data_path + "val_with_commonsense_final.json","w") as f:
    #     json.dump(val,f)
        
    # # Test
    # for key,value in tqdm(test.items(), desc="Test"):
    #     try:
    #         topic = value["post"]
    #         if len(topic) > 50:
    #             topic = topic[0:50]
    #         outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         # outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         # try:
    #         #     outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         # except Exception as e:
    #         #     topic = topic[0:min(50,len(topic))]
    #         #     outputs = interactive.get_atomic_sequence(topic, model, sampler, data_loader, text_encoder, category)
    #         # print(topic)
    #         # print("------------------------------")
    #         # print(outputs)
    #         # print("----------------------------------")
    #         value["commonsense"] = outputs
    #     except Exception as e:
    #         print(e)
    #         print(key)
    # with open(data_path + "test_with_commonsense_final.json","w") as f:
    #     json.dump(test,f)