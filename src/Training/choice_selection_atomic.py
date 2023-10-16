import argparse
import os
import pandas as pd
import random
import numpy as np
from ast import literal_eval
def main():
    parser = argparse.ArgumentParser()
    #shared argument group
    parser.add_argument("--input_path", type=str, required=True, help="The path of the mean_var_td")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to output the new data")
    parser.add_argument("--number_of_negative", type=int, default=1, help="The number of negative example to put in h")
    parser.add_argument("--num_cand", type=int, default=5, help="number of choices")
    parser.add_argument("--metrics", type=str, default="mean", help='metrics to extract choices')
    parser.add_argument("--high", action='store_true', default=False, help='choose choice with high value if store_true')
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    
    num_choices = args.number_of_negative + 1
    df = pd.read_json(args.input_path,lines=True)
    metrics = np.array(df[args.metrics].tolist())
    choices = np.array(df["candidates"].tolist())
    filename = args.output_dir+'/'+args.name+"_"+args.metrics+'_'+str(args.high)+'_'+"top"+str(num_choices)
    labels = np.zeros((choices.shape[0]),dtype=int)
    if args.high:
        metrics[[a for a in range(metrics.shape[0])],[int(i) for i in df['correct'].tolist()]] = 1.0
        metrics = np.argsort(-metrics,axis=1)
    else:
        metrics[[a for a in range(metrics.shape[0])],[int(i) for i in df['correct'].tolist()]] = 0.0
        metrics = np.argsort(metrics,axis=1)
        
    for i in range(num_choices):
        temp = choices[[a for a in range(choices.shape[0])],[b for b in metrics[:,i]]].reshape(-1,1)
        if i == 0:
            result = temp
        else:
            result = np.hstack((result,temp))
    output = pd.DataFrame({"id": df['id'],
                        "dim": df['dim'],
                        "context": df['context'],
                        "correct": labels,
                        "candidates": list(result),
                        "keywords": df['keywords'],
                        'mean':df['mean'],
                        'var':df['var'],
                        'q_mean':df['q_mean'],
                        'q_var':df['q_var'],
                        'fn':df['fn']})
    output.to_json(filename,lines=True,orient="records")
    print("Data has been stored in "+filename)
    




if __name__ == "__main__":
    main()