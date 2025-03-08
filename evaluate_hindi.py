#!/usr/bin/env python

import sys
import os
import pandas as pd
import eval_metrics_DF as em
import numpy as np

if len(sys.argv) != 4:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    exit(1)

submit_file = sys.argv[1]  # Path to scores file
truth_dir = sys.argv[2]   # Directory containing keys file
phase = sys.argv[3]

cm_key_file = os.path.join(truth_dir, 'cmkeyhindi.txt')

def eval_to_score_file(score_file, cm_key_file):
    cm_data = pd.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pd.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        print(f'CHECK: submission has {len(submission_scores)} of {len(cm_data)} expected trials.')
        exit(1)

    if len(submission_scores.columns) > 2:
        print(f'CHECK: submission has more columns ({len(submission_scores.columns)}) than expected (2).')
        exit(1)

    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=0, how='inner')
    print(cm_scores.columns)

    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    # eer_cm = em.compute_eer(np.array([0]), spoof_cm)[0]  # Only spoof scores provided

    out_data = f"eer: {100 * eer_cm:.2f}\n"
    print(out_data)
    return eer_cm

if __name__ == "__main__":

    if not os.path.isfile(submit_file):
        print(f"{submit_file} doesn't exist")
        exit(1)

    if not os.path.isdir(truth_dir):
        print(f"{truth_dir} doesn't exist")
        exit(1)

    if phase not in ('progress', 'eval', 'hidden_track'):
        print("phase must be either progress, eval, or hidden_track")
        exit(1)

    _ = eval_to_score_file(submit_file, cm_key_file)
