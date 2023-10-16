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
    metrics = np.array(df['mean'].tolist())
    filename = args.output_dir+'/'+args.name+"_"+args.metrics+'_'+str(args.high)+'_'+"top"+str(num_choices)
    labels = ['A']*int(df.shape[0])
    labels = []
    for i in range(int(df.shape[0])):
        if df['answerKey'][i] == 'A':
            labels.append(0)
        elif df['answerKey'][i] == 'B':
            labels.append(1)
        elif df['answerKey'][i] == 'C':
            labels.append(2)
    print(metrics.shape[0])
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            if metrics[i][j]==None:
                print("error "+str(i)+"  "+str(j))
    if args.high:
        metrics[[a for a in range(metrics.shape[0])],[int(i) for i in labels]] = 1.0
        metrics = np.argsort(-metrics,axis=1)
    else:
        metrics[[a for a in range(metrics.shape[0])],[int(i) for i in labels]] = 0.0
        metrics = np.argsort(metrics,axis=1)
    
    for i in range(int(df.shape[0])):
        df['question'][i]['choices'][metrics[i][0]]['label'] = 'A'
        df['question'][i]['choices'][metrics[i][1]]['label'] = 'B'
        df['question'][i]['choices'] = [df['question'][i]['choices'][metrics[i][0]],df['question'][i]['choices'][metrics[i][1]]]
        df['answerKey'][i] = 'A'
        

    df.to_json(filename,lines=True,orient="records")
    print("Data has been stored in "+filename)
    
if __name__ == "__main__":
    main()