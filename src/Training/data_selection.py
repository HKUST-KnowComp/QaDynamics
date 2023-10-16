import argparse
import json
import logging
import numpy as np
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    #shared argument group
    parser.add_argument("--input_path", type=str, required=True, help="The path of the mean_var_td")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to output the new data")
    parser.add_argument("--metrics", type=str, required=True,
                        choices=("var",
                                 "mean",
                                 "cor",
                                 "random"),
                        help="The metric of data to perform selection.")
    parser.add_argument("--high", action='store_true', default=False, help="Choose the data with high metrics or low metrics.")
    parser.add_argument("--percent", type=float, default=0.33, help="The percent of data need to be selected.")
    parser.add_argument("--selection", type=str, default='question',
                        choices=('question',
                                 'choices'),
                        help="When performing data selection, select the question to generate a new dataset or select subset of choices")
    parser.add_argument("--seed", type=int,default=400,help="seed")
    #parser.add_argument("--amount", type=int,default=400,help="amount of data")
    #argument for choices selection
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = pd.read_json(args.input_path, lines=True)
    if args.selection == 'question':
        if args.metrics!='random':
            if args.high:
                df = df.sort_values(by='q_'+args.metrics, ascending=False)
            else:
                df = df.sort_values(by='q_'+args.metrics)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            file_name = args.output_dir+'/'+args.selection+'_'+args.metrics+str(args.percent)+str(args.high)+".jsonl"
            result = df[:int(df.shape[0]*args.percent)]
            #result = df[:args.amount]
            result.to_json(file_name,orient='records',lines=True)
            print("file has been successfully stored in "+file_name)
        else:
            result = df.sample(frac=args.percent)
            file_name = args.output_dir+'/'+args.selection+'_'+args.metrics+str(args.percent)+str(args.seed)+".jsonl"
            result.to_json(file_name,orient='records',lines=True)
            print("file has been successfully stored in "+file_name)

if __name__ == "__main__":
    main()