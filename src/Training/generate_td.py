import os
import argparse
import numpy as np
import pandas as pd
import logging
from typing import List
import torch
import torch.nn.functional as F
from ast import literal_eval

def log_mean_var(output_dir: os.path,
                 exp_name: str,
                 original_data,
                 mean: List[List[float]],
                 var: List[List[float]],
                 q_mean: List[float],
                 q_var: List[float],
                 fn: List[float]):
    training_dynamics = pd.DataFrame({
        "mean": mean,
        "var": var,
        "q_mean": q_mean,
        "q_var": q_var,
        "fn": fn
    })
    td_df = pd.concat([original_data, training_dynamics],axis=1)
    logging_dir = output_dir
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    file_name = os.path.join(logging_dir, "mean_var_td_"+exp_name+".jsonl")
    td_df.to_json(file_name,orient='records',lines=True)
    print("The file has been successfully stored in "+file_name)



def main():
    #argument of confidence and variance
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic_path", type=str, required=False, help="The path of atomic qa")
    parser.add_argument("--cwwv_path", type=str, required=False, help="The path of cwwv qa")
    parser.add_argument("--input_dir", type=str, required=True, help="The directory to store the training dynamics")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to output the data with mean and variance")
    parser.add_argument("--start", type=int, required=True, help="Number of epochs you want to involve.")
    parser.add_argument("--end", type=int, required=True, help="Number of epochs you want to involve.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--margin",action="store_true",default=True,help="Whether to use margin to measure the confidence of question")
    args = parser.parse_args()

    atomic_qa = pd.read_json(args.atomic_path, lines=True)
    cwwv_qa = pd.read_json(args.cwwv_path, lines=True)
    cwwv_label = []
    for l in cwwv_qa['answerKey']:
        if l == 'A':
            cwwv_label.append(0)
        elif l == 'B':
            cwwv_label.append(1)
        elif l == 'C':
            cwwv_label.append(2)
    atomic_label = atomic_qa['correct'].tolist()
    labels = cwwv_label+atomic_label
    logits = np.array([])
    one_hots = np.array([]) #temp array for computation of confidence and variance
    question_mean = np.array([])
    question_var = np.array([])
    num_cand = 3
    for i in range(args.start, args.end):
        df = pd.read_csv(args.input_dir+"logits_test.txt_"+str(float(i)), sep=' ',header=None)
        logit = df.to_numpy()
        logit[np.isnan(logit)] = -100.0
        length = df.shape[0]
        one_hot = np.zeros((length, num_cand),dtype=np.float32)
        one_hot[[a for a in range(length)],[int(i) for i in labels]] = 1.0
        if i == args.start:
            logits = np.array(logit).reshape(1,-1)
            one_hots = one_hot
        else:
            logits=np.concatenate((logits, logit.reshape(1,-1)), axis=0)
            one_hots = np.concatenate((one_hots, one_hot), axis=0)
    print("The mean var computation has finished!")
    fn = np.mean(logits, axis=0, keepdims=True)
    fn = fn.reshape(-1, num_cand) # num_question*num_cand
    incorrect_logit = np.copy(fn)
    incorrect_logit[[a for a in range(length)],[int(i) for i in labels]] = -150
    incorrect_logit = np.max(incorrect_logit, axis=1)
    f_negative = np.absolute(fn[[a for a in range(length)],[int(i) for i in labels]] - incorrect_logit) #num_question
    

    logits = logits.reshape(-1, num_cand)
    one_hots = one_hots.reshape(-1, num_cand) # num_question*num_cand
    logits = torch.tensor(logits, dtype=torch.float)
    '''index = np.argmax(one_hots, axis=1).tolist()
    logits = torch.tensor(logits, dtype=torch.float)
    correct = logits.clone()
    second_wrong = logits.clone()
    correct = correct[[i for i in range(logits.shape[0])],[j for j in index]].clone().reshape(1,-1)
    second_wrong[[i for i in range(logits.shape[0])],[j for j in index]] = -100 #-inf
    second_wrong = torch.topk(second_wrong, 2, dim=1).values[:,-1].reshape(1, -1)
    prob = F.softmax(logits,dim=1)
    prob[[i for i in range(logits.shape[0])],[j for j in index]] = torch.softmax(torch.cat([correct,second_wrong],dim=0), dim=0)[0]
    prob = prob.numpy() #The numpy array storing the softmax probability'''
    prob = F.softmax(logits,dim=1).numpy()
    temp = 1.0-np.absolute(prob-one_hots).reshape(args.end-args.start, -1) # (start-end*num_question)*num_cand
    #temp = logits
    mean = np.mean(temp, axis=0).reshape(-1, num_cand).tolist()
    var = np.std(temp, axis=0).reshape(-1, num_cand).tolist()
    #mean,var shape: 691551 * 3
    if args.margin:
        #use margin to present the confidence and variance of question
        one_hots = one_hots*3-1
        prob = prob*one_hots
        prob = np.mean(prob, axis=1, keepdims=True) #(args.start-args.end*num_question)*1
        prob = prob.reshape(args.end-args.start, -1)
        question_mean = np.mean(prob, axis=0).tolist()
        question_var = np.std(prob, axis=0).tolist()
    else:
        #use average to present the confidence and variance of question
        temp = 1 - np.absolute(prob - one_hots)
        temp = np.mean(temp, axis=1, keepdims=True).reshape(args.end-args.start, -1)
        question_mean = np.mean(temp, axis=0).tolist()
        question_var = np.std(temp, axis=0).tolist()
    label_df = pd.DataFrame({
        "labels": labels
    })
    log_mean_var(
        args.output_dir,
        args.exp_name+"graph",
        label_df,
        mean,
        var,
        question_mean,
        question_var,
        f_negative
    )
    #log mean and var to original data atomic
    log_mean_var(
        args.output_dir,
        args.exp_name+"atomic",
        atomic_qa, 
        mean[len(cwwv_qa):],
        var[len(cwwv_qa):],
        question_mean[len(cwwv_qa):],
        question_var[len(cwwv_qa):],
        f_negative[len(cwwv_qa):]
    )
    #log mean and var to original data cwwv
    log_mean_var(
        args.output_dir,
        args.exp_name+"cwwv",
        cwwv_qa, 
        mean[0:len(cwwv_qa)],
        var[0:len(cwwv_qa)],
        question_mean[0:len(cwwv_qa)],
        question_var[0:len(cwwv_qa)],
        f_negative[0:len(cwwv_qa)]
    )

if __name__ == "__main__":
    main()