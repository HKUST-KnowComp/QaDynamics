import argparse
import json
import logging
import numpy as np
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    #shared argument group
    parser.add_argument("--atomic_path", type=str, required=True, help="The path of the atomic")
    parser.add_argument("--cwwv_path", type=str, required=True, help="The path of cwwv")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to output the new data")
    parser.add_argument("--strategy", type=int, required=True, help="adopt which strategy, 1 is strategy 1, 2 is strategy, 3 is combination")
    parser.add_argument("--threshold1", type=float, required=True, help="Threshold for strategy 1")
    parser.add_argument("--threshold2", type=float, required=True, help="Threshold for strategy 2")
    parser.add_argument("--name", type=str, required=True, help="output filename")
    parser.add_argument("--seed", type=int,default=400,help="seed")
    #argument for choices selection
    args = parser.parse_args()
    df_atomic = pd.read_json(args.atomic_path, lines=True)
    df_cwwv = pd.read_json(args.cwwv_path, lines=True)
    print("Data length atomic and cwwv is {} {}.".format(len(df_atomic), len(df_cwwv)))
    if args.strategy==1:
        for i in range(len(df_atomic)):
            if df_atomic['mean'][i][df_atomic['correct'][i]]<args.threshold1:
                df_atomic['correct'][i]+=3
        df_atomic_1 = df_atomic.drop(df_atomic[df_atomic['correct']>=3].index)
        for i in range(len(df_cwwv)):
            temp = 0
            if df_cwwv['answerKey'][i]=='A':
                temp=0
            elif df_cwwv['answerKey'][i]=='B':
                temp=1
            elif df_cwwv['answerKey'][i]=='C':
                temp=2
            if df_cwwv['mean'][i][temp]<args.threshold1:
                df_cwwv['answerKey'][i]=-1
        df_cwwv_1 = df_cwwv.drop(df_cwwv[df_cwwv['answerKey']==-1].index)
        print("Data length of updated atomic and cwwv is {} {}.".format(len(df_atomic_1), len(df_cwwv_1)))
        df_atomic_1.to_json(args.output_dir+"/"+args.name+"atomic1"+".jsonl",lines=True, orient="records")
        df_cwwv_1.to_json(args.output_dir+"/"+args.name+"cwwv1"+".jsonl", lines=True, orient="records")
    elif args.strategy==2:
        df_atomic_2 = df_atomic.drop(df_atomic[df_atomic["fn"]<args.threshold2].index)
        df_cwwv_2 = df_cwwv.drop(df_cwwv[df_cwwv['fn']<args.threshold2].index)
        print("Data length of cleared atomic is "+str(len(df_atomic_2)))
        print("Data length of cleared cwwv is "+str(len(df_cwwv_2)))
        df_atomic_2.to_json(args.output_dir+"/"+args.name+"atomic2"+".jsonl",lines=True, orient="records")
        df_cwwv_2.to_json(args.output_dir+"/"+args.name+"cwwv2"+".jsonl", lines=True, orient="records")
    elif args.strategy==3:
        for i in range(len(df_atomic)):
            if df_atomic['mean'][i][df_atomic['correct'][i]]<args.threshold1:
                df_atomic['correct'][i]+=3
        df_atomic_1 = df_atomic.drop(df_atomic[df_atomic['correct']>=3].index)
        for i in range(len(df_cwwv)):
            temp = 0
            if df_cwwv['answerKey'][i]=='A':
                temp=0
            elif df_cwwv['answerKey'][i]=='B':
                temp=1
            elif df_cwwv['answerKey'][i]=='C':
                temp=2
            if df_cwwv['mean'][i][temp]<args.threshold1:
                df_cwwv['answerKey'][i]=-1
        df_cwwv_1 = df_cwwv.drop(df_cwwv[df_cwwv['answerKey']==-1].index)
        df_atomic_2 = df_atomic_1.drop(df_atomic_1[df_atomic_1["fn"]<args.threshold2].index)
        df_cwwv_2 = df_cwwv_1.drop(df_cwwv_1[df_cwwv_1['fn']<args.threshold2].index)
        print("Data length of updated atomic and cwwv is {} {}.".format(len(df_atomic_2), len(df_cwwv_2)))
        df_atomic_2.to_json(args.output_dir+"/"+args.name+"atomic3"+".jsonl",lines=True, orient="records")
        df_cwwv_2.to_json(args.output_dir+"/"+args.name+"cwwv3"+".jsonl", lines=True, orient="records")
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()