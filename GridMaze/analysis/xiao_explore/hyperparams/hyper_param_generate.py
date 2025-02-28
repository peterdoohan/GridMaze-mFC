import csv
import pandas as pd
import shutil


def main():
    maze_ids = list(range(1, 4))
    subject_IDs = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
    sessions = list(range(13))
    combos = [(maze_id , subject_ID, session) for maze_id in maze_ids for subject_ID in subject_IDs for session in sessions]
    df = pd.DataFrame(combos, columns=['maze_id', 'subject_ID', 'session'])
    df.to_csv('/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/hyperparams/hyperparam_basic/maze_subject_session.txt', index=False, sep='\t')
    shutil.copy(__file__, '/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/hyperparams/hyperparam_basic/maze_subject_session_generate.py' )
    return


if __name__ == "__main__":
    main()