""" """
#%%Imports
import json
import copy
import numpy as np
import pandas as pd
import maze_analysis
import seaborn as sns
# from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %% Load data for analysis 

with open('../data/experiment_info.json') as input_file:
    exp_info = json.load(input_file)
    
import analysis.get_sessions as gs
sessions = gs.load_sessions('../data')
grouped_sessions_df = gs.get_grouped_sessions_df(sessions)
del sessions

import analysis.decoding_analysis as decoding_analysis
valid_sessions_df = decoding_analysis.get_valid_grouped_sessions_df(grouped_sessions_df, exp_info)
del grouped_sessions_df

decoding_kwargs = {'event':'reward', 'start_win': -0.5, 'end_win':0.5}
valid_sessions_df['activity_df'] = valid_sessions_df['sessions'].apply(lambda x: decoding_analysis.get_activity_df(x, **decoding_kwargs))

# %% Goal Representation Similarity Regression

def get_goal_RS_analysis_df(valid_sessions_df, exp_info, regressors, previous_maze_regressors=False):
    """input must be valid group sessions """
    info2weights = []
    maze2goals = exp_info['maze_day2goals']
    subject_IDs = exp_info['subject_IDs']
    subject_IDs = [s for s in subject_IDs if s!='m3']
    for m, maze in enumerate(maze2goals.keys()):
        maze_mask = valid_sessions_df['maze'] == m+1
        goal_sets = np.unique([x for x in maze2goals[maze].values()])
        for goal_set in goal_sets:
            goal_set_mask = valid_sessions_df['goal_set']==goal_set
            subject_goal_RSMs = []
            for subject in exp_info['subject_IDs']:
                subject_mask = valid_sessions_df['subject_ID'] == subject
                try:
                    activity_df = valid_sessions_df[subject_mask & maze_mask & goal_set_mask]['activity_df'].to_list()[0]
                    subject_goal_RSMs.append(get_goal_representation_similarity_matrix(activity_df))
                except IndexError: #no valid subject sessions
                    pass
            av_goal_RSM = np.mean(subject_goal_RSMs, axis=0)
            example_session = valid_sessions_df[maze_mask & goal_set_mask]['sessions'].iloc[-1][0]
            maze_name = f'maze {example_session.maze_number}'
            goal_coords = maze_analysis._get_node_coords(example_session.goals)
            regressor2weight = regress_goal_RS_with_maze_features(maze_name, goal_coords, av_goal_RSM, exp_info, 
                                                                  regressors, previous_maze_regressors=previous_maze_regressors)
            metadata = {'maze': m+1, 'goal_set': goal_set}
            info2weights.append({**metadata, **regressor2weight})
    return pd.DataFrame(info2weights)


def regress_goal_RS_with_maze_features(maze_name, goal_coords, av_goal_RSM, exp_info, regressors, previous_maze_regressors=False):
    """Adds maze features for a other mazes as coregressors """
    # set up mazes for coregressors
    maze_name2nx_maze = maze_analysis.get_nx_mazes_dict(exp_info)
    maze_names = [i for i in maze_name2nx_maze.keys()]
    nx_mazes = [i for i in maze_name2nx_maze.values()]
    current_nx_maze = maze_name2nx_maze[maze_name]
    if previous_maze_regressors:
        for i, mn in enumerate(maze_names):
            if maze_name == maze_names[i]: 
                maze_names_for_regression = maze_names[:(i+1)]
                nx_mazes_for_regression = nx_mazes[:(i+1)]
    else: 
        maze_names_for_regression = [maze_name]
        nx_mazes_for_regression = [current_nx_maze]
    # add regressions to the input matrix
    metric2normalised = maze_analysis.get_metric2normalised_dict()
    input_matrix = None
    regressor2weight = {}
    for reg in regressors:
        if reg == 'euclidean':
            euclidean_vector = _get_similarity_vector(current_nx_maze, goal_coords, exp_info, regressor=reg, normalise=True)
            input_matrix, regressor2weight = _update_input_matrix(input_matrix, regressor2weight, input_vector=euclidean_vector, regressor=reg)
        else:
            if 'ortho' in reg.split('_'):
                ortho_reg = reg.split('_', 1)[-1]
                normalise_reg = not metric2normalised[ortho_reg]
                euclidean_vector = _get_similarity_vector(current_nx_maze, goal_coords, exp_info, regressor='euclidean', normalise=True)
                reg_vectors = np.array([_get_similarity_vector(nx_maze, goal_coords, exp_info, regressor=ortho_reg, normalise=normalise_reg) 
                                        for nx_maze in nx_mazes_for_regression])
                vectors_for_orthog = np.vstack((euclidean_vector, *reg_vectors[:-1]))
                
                ortho_reg_vectors = [gram_schmidt(reg_vectors[i], vectors_for_orthog[:i+1]) for i in range(len(reg_vectors))]
                for maze_name, ortho_reg_vector in zip(maze_names_for_regression, ortho_reg_vectors):
                    if previous_maze_regressors: reg_name = f'{maze_name} '+reg
                    else: reg_name = reg
                    input_matrix, regressor2weight = _update_input_matrix(input_matrix, regressor2weight, input_vector=ortho_reg_vector, 
                                                                     regressor=(reg_name))
            else: # process without orthogonalisation
                normalise_reg = not metric2normalised[reg]
                reg_vectors = [_get_similarity_vector(nx_maze, goal_coords, exp_info, regressor=reg, normalise=~normalise_reg) 
                               for nx_maze in nx_mazes_for_regression]
                for maze_name, reg_vector in zip(maze_names_for_regression, reg_vectors):
                    if previous_maze_regressors: reg_name = f'{maze_name} '+reg
                    else: reg_name = reg
                    input_matrix, regressor2weight = _update_input_matrix(input_matrix, regressor2weight, input_vector=reg_vector, 
                                                                     regressor=(reg_name))
    goal_RSM_upper = get_upper_triangle(av_goal_RSM)  
    if len(input_matrix.shape) == 1: #reshape if only one regressor
        input_matrix = input_matrix.reshape(1,-1).T
        goal_RSM_upper = goal_RSM_upper.reshape(1,-1).T
    linreg = LinearRegression(fit_intercept=True).fit(input_matrix, goal_RSM_upper)
    beta_coefs = linreg.coef_
    if len(beta_coefs)==1: beta_coefs = beta_coefs[0] #if only one regressor
    for reg, idx in regressor2weight.items():
        regressor2weight[reg] = beta_coefs[idx]
    return regressor2weight


# %% Control Check autocorrelation between maze feature regressors 


def check_autocorrelation_between_maze_metrics():
    """Will not work - need to update get_similarity_matrix to new inputs """
# get example sessions for each mazes
    m1_session = valid_sessions_df[valid_sessions_df['maze']==1]['sessions'].iloc[0][0]
    m2_session = valid_sessions_df[valid_sessions_df['maze']==2]['sessions'].iloc[2][0]
    m3_session = valid_sessions_df[valid_sessions_df['maze']==3]['sessions'].iloc[4][0]
    
    cross_corrs = []
    for i, session in enumerate([m1_session, m2_session, m3_session]):
        euc = maze_analysis.get_similarity_matrix(session, exp_info, function_type='euclidean')
        euc = maze_analysis.normalise_similarity_matrix(euc)
        euc = get_upper_triangle(euc)
        metrics = [i for i in maze_analysis.get_metric2normalised_dict().keys()]
        for metric in metrics:
            met = maze_analysis.get_similarity_matrix(session, exp_info, function_type=metric)
            if not maze_analysis.get_metric2normalised_dict()[metric]:
                met = maze_analysis.normalise_similarity_matrix(met)
            met = get_upper_triangle(met)
            corr = np.min(np.corrcoef(euc, met))
            cross_corrs.append({'maze': i+1, 'metric': metric, 'corr': corr})
    cross_corr_df = pd.DataFrame(cross_corrs)
    return cross_corr_df
        

# %% bootstap get_goal_RS_analysis for statistical tests

def get_bootstrapped_goal_RS_analysis_dfs(valid_sessions_df, exp_info, regressors, n=1000, previous_maze_regressors=False):
    """input must be valid group sessions """
    goal_RSA_dfs = []
    for i in range(n):
        try:
            if i%100 == 0: print(i)
            info2weights = []
            maze2goals = exp_info['maze_day2goals']
            subject_IDs = exp_info['subject_IDs']
            subject_IDs = [s for s in subject_IDs if s!='m3']
            permuted_subjects = np.random.choice(subject_IDs, len(subject_IDs))
            for m, maze in enumerate(maze2goals.keys()):
                maze_mask = valid_sessions_df['maze'] == m+1
                goal_sets = np.unique([x for x in maze2goals[maze].values()])
                for goal_set in goal_sets:
                    goal_set_mask = valid_sessions_df['goal_set']==goal_set
                    subject_goal_RSMs = []
                    for subject in permuted_subjects:
                        subject_mask = valid_sessions_df['subject_ID'] == subject
                        try:
                            activity_df = valid_sessions_df[subject_mask & maze_mask & goal_set_mask]['activity_df'].to_list()[0]
                            subject_goal_RSMs.append(get_goal_representation_similarity_matrix(activity_df))
                        except IndexError: #no valid subject sessions
                            pass
                    av_goal_RSM = np.mean(subject_goal_RSMs, axis=0)
                    example_session = valid_sessions_df[maze_mask & goal_set_mask]['sessions'].iloc[-1][0]
                    
                    av_goal_RSM = np.mean(subject_goal_RSMs, axis=0)
                    example_session = valid_sessions_df[maze_mask & goal_set_mask]['sessions'].iloc[-1][0]
                    maze_name = f'maze {example_session.maze_number}'
                    goal_coords = maze_analysis._get_node_coords(example_session.goals)
                    regressor2weight = regress_goal_RS_with_maze_features(maze_name, goal_coords, av_goal_RSM, exp_info, 
                                                                          regressors, previous_maze_regressors=previous_maze_regressors)
                    metadata = {'iter':i, 'maze': m+1, 'goal_set': goal_set}
                    info2weights.append({**metadata, **regressor2weight})
            goal_RSA_dfs.append(pd.DataFrame(info2weights)) #no valid 
        except IndexError: #no valid sessions for every av_goal_RSM
            pass
    return  goal_RSA_dfs


# %% Goal Representation Similarity Regression Subfunctions

def get_average_goal_RSM(valid_sessions_df, exp_info, subject_IDs, maze_no, goal_sets):
    if subject_IDs == 'all': subject_IDs = exp_info['subject_IDs']
    if maze_no == 'all': maze_no = np.arange(1,len(exp_info['maze_config2info'].keys())+1)
    if goal_sets == 'all': 
        maze_goal_sets = [[i for i in exp_info['maze_day2goals'][key].values()] for key in exp_info['maze_day2goals']]
        unique_goal_sets = np.unique([g_set for inner_list in maze_goal_sets for g_set in inner_list])
        goal_sets = unique_goal_sets
    subject_goal_RSMs = []
    for subject in subject_IDs:
        subject_mask = valid_sessions_df['subject_ID'] == subject
        for maze in maze_no:
            maze_mask = valid_sessions_df['maze'] == maze
            for goal_set in goal_sets:
                goal_set_mask = valid_sessions_df['goal_set']==goal_set
                try:
                    activity_df = valid_sessions_df[subject_mask & maze_mask & goal_set_mask]['activity_df'].to_list()[0]
                    subject_goal_RSMs.append(get_goal_representation_similarity_matrix(activity_df))
                except IndexError: #no valid subject sessions
                    pass
    av_goal_RSM = np.mean(subject_goal_RSMs, axis=0)
    return av_goal_RSM


def get_goal_representation_similarity_matrix(activity_df):
    """Finds the pearson correlations between goal representations in an activity_df (single window)"""
    grouped_activity_df = activity_df.groupby(['goal', 'cluster_unique_ID']).mean()['activity']
    goals = np.sort(activity_df['goal'].unique())
    activity_vectors = [grouped_activity_df[goal].to_numpy() for goal in goals]
    return np.corrcoef(activity_vectors)  


def _get_similarity_vector(maze, goal_coords, exp_info, regressor, normalise=True):
    similarity_matrix = maze_analysis.get_similarity_matrix(maze, goal_coords, exp_info, function_type=regressor)
    if normalise:
        similarity_matrix = maze_analysis.normalise_similarity_matrix(similarity_matrix)
    return get_upper_triangle(similarity_matrix)


def orthogonal_projection(vector1, vector2):
    """Finds the orthogonal projection of vector2 into vector1"""
    return vector2 - np.dot(vector2, vector1) / np.dot(vector1, vector1)*vector1

def gram_schmidt(v, w):
    """Orthogonalize a vector v to n other orthogonal vectors in w."""
    v = np.array(v)
    if len(w.shape) == 1:
        v = v - np.dot(v, w) / np.dot(w, w) * w
    else:
        for u in w:
            v = v - np.dot(v, u) / np.dot(u, u) * u
    return v


def get_upper_triangle(matrix):
    row_indices, col_indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_indices, col_indices]


def _update_input_matrix(input_matrix, beta_values, input_vector, regressor):
    if input_matrix is None:
        input_matrix = input_vector
    else:
        input_matrix = np.column_stack((input_matrix, input_vector))
    beta_values[regressor] = len(beta_values)
    return input_matrix, beta_values


# %% RSA regression Anylsis and plotting

def plot_RSA_results(valid_sessions_df, exp_info, regressors, previous_maze_regressors=False):
    goal_RSA_df = get_goal_RS_analysis_df(valid_sessions_df, exp_info, regressors, 
                                          previous_maze_regressors=previous_maze_regressors).groupby(['maze']).mean().reset_index()
    CIs_df = get_CIs_df(valid_sessions_df, exp_info, regressors, n_bs=100, previous_maze_regressors=previous_maze_regressors)
    # plotting 
    fig, ax = plt.subplots(figsize=(4,6))
    mazes = goal_RSA_df['maze']
    x = 0.5*np.arange(len(mazes))
    x_offset = 0.03
    regressor_names = [i for i in goal_RSA_df.columns[1:]]
    for r, reg in enumerate(regressor_names):
        spacings = x + r*x_offset
        betas = goal_RSA_df[reg]
        errors_df = CIs_df[CIs_df['regressor']==reg]
        error_lower = errors_df['abs_error_lower']
        error_upper = errors_df['abs_error_upper']
        ax.errorbar(spacings, betas, yerr=[error_lower, error_upper], 
                    label=reg, marker='o', markersize=10, linestyle='None', capsize=4, elinewidth=2, capthick=2)
    ax.axhline(color='k', linestyle='--')
    # Add labels and title
    ax.set_ylabel('Regressor Weights')
    ax.set_xticks(x+x_offset*len(regressor_names))
    ax.set_xticklabels([f'Maze {i}' for i in mazes])
    ax.set_ylim(-0.2,0.2)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def get_CIs_df(valid_sessions_df, exp_info, regressors, n_bs = 1000, previous_maze_regressors=False):
    bootstrapped_goal_RSA_regressions = get_bootstrapped_goal_RS_analysis_dfs(valid_sessions_df, exp_info, regressors, n=n_bs,
                                                                              previous_maze_regressors=previous_maze_regressors)
    bootstrapped_df = pd.concat([x.groupby(x['maze']).mean() for x in bootstrapped_goal_RSA_regressions]).reset_index(drop=False)
    m1_mask = bootstrapped_df['maze']==1
    m2_mask = bootstrapped_df['maze']==2
    m3_mask = bootstrapped_df['maze']==3
    bootstrap_CIs = []
    for x, maze_mask in enumerate([m1_mask, m2_mask, m3_mask]):
        regressor_names = [i for i in bootstrapped_df.columns[2:]]
        for reg in regressor_names:
            bootstrapped_reg = bootstrapped_df[maze_mask][reg] #error here
            mean_reg = np.mean(bootstrapped_reg)
            CI95_lower = np.percentile(bootstrapped_reg, 2.5)
            CI95_upper = np.percentile(bootstrapped_reg, 97.5)
            bootstrap_CIs.append({'maze': x+1, 'regressor': reg, 'abs_error_lower':mean_reg - CI95_lower , 'abs_error_upper': CI95_upper - mean_reg})   
    return pd.DataFrame(bootstrap_CIs)
# %% plots for a set of regressors

regressor_pairs = [['euclidean', 'ortho_geodesic'], 
                  ['euclidean', 'ortho_connectivity_similarity'], 
                  ['euclidean', 'ortho_same_room'],
                  ['euclidean', 'ortho_path_length_similarity'], 
                  ['euclidean', 'ortho_degree_similarity'], 
                  ['euclidean', 'ortho_geodesic_minus_euclidean']]
for regressor_pair in regressor_pairs:
    regressors = regressor_pair
    plot_RSA_results(valid_sessions_df, exp_info, regressors, previous_maze_regressors=False)
    plot_RSA_results(valid_sessions_df, exp_info, regressors, previous_maze_regressors=True)
    

# %% Within Subject RSA Analysis

def correlate_subject_GRS_within_mazes(valid_sessions_df, exp_info):
    """Correlates goal representation similarity (GRS) for a single subject with the average GRS of all other subjects on a given maze"""
    info2corr = []
    decoding_kwargs = {'event':'reward', 'start_win': -0.5, 'end_win':0.5}
    subject_IDs = exp_info['subject_IDs']
    maze2subset = exp_info['maze_day2goals']
    maze_keys = list(maze2subset.keys())
    goal_sets = np.unique([x for x in maze2subset['maze 1'].values()]) #same across mazes
    for subject in subject_IDs:
        for m, maze in enumerate(maze_keys):
            for goal_set in goal_sets:
                try:
                    #activity corr for held out subject
                    subject_RSM = _get_subject_RSM(valid_sessions_df, decoding_kwargs, subject=subject, maze_no=m, goal_set=goal_set)
                    subject_RSMupper = get_upper_triangle(subject_RSM)
                    print(f'Correlating subject {subject}, {maze}, {goal_set} within mazes')
                    # av activity cor for other subjects
                    other_subjects = copy.deepcopy(subject_IDs) #deepcopy to not alter exp_info dict
                    other_subjects.remove(subject)
                    other_RSMs = []
                    for other in other_subjects:
                        try:
                            other_RSM = _get_subject_RSM(valid_sessions_df, decoding_kwargs, subject=other, maze_no=m, goal_set=goal_set)
                            other_RSMs.append(other_RSM)
                        except ValueError: #no valid sessions for this (other) subject, maze, goal_set combination
                            print(f'Failed to encorporate subject {other}, {maze}, {goal_set} into others_RSM (no valid sessions)')
                            pass
                    others_RSM = np.mean(other_RSMs, axis=0)
                    others_RSMupper = get_upper_triangle(others_RSM)
                    # corr of subject with av of others    
                    corr = np.min(np.corrcoef(subject_RSMupper, others_RSMupper))
                    info2corr.append({'subject': subject, 'maze': maze, 'goal_set': goal_set,
                                      'correlation': corr})
                except IndexError: #error if sub,maze,goal_set is not in df because no valid sessions
                    print(f'Failed to assess {subject}, {maze}, {goal_set} (no valid sessions)')
                    pass
    return pd.DataFrame(info2corr)



def correlate_subject_GRS_across_mazes(valid_sessions_df, exp_info):
    """Correlates goal representation similarity (GRS) for a single subject on a given maze with the average GRS of all other subjects on a different maze"""
    info2corr = []
    decoding_kwargs = {'event':'reward', 'start_win': -0.5, 'end_win':0.5}
    maze2subset = exp_info['maze_day2goals']
    subject_IDs = exp_info['subject_IDs']
    maze_keys = list(maze2subset.keys())
    goal_sets = np.unique([x for x in maze2subset['maze 1'].values()]) #same across mazes
    for subject in subject_IDs:
        for sm, subject_maze in enumerate(maze_keys):
            for om, others_maze in enumerate(maze_keys[sm+1:]):
                for goal_set in goal_sets:
                    try:
                        # RSM subject
                        subject_RSM = _get_subject_RSM(valid_sessions_df, decoding_kwargs, subject=subject, maze_no=sm, goal_set=goal_set)
                        subject_RSMupper = get_upper_triangle(subject_RSM)
                        print(f'Correlating subject {subject}, {subject_maze}, {goal_set} across mazes')
                        # RSM others
                        other_subjects = copy.deepcopy(subject_IDs) #deepcopy to not alter exp_info dict
                        other_subjects.remove(subject)
                        other_RSMs = []
                        for other in other_subjects:
                            try:
                                other_RSM = _get_subject_RSM(valid_sessions_df, decoding_kwargs, subject=other, maze_no=sm+1, goal_set=goal_set)
                                other_RSMs.append(other_RSM)
                            except ValueError: #no valid sessions for this (other) subject, maze, goal_set combination
                                print(f'Failed to encorporate subject {other}, {others_maze}, {goal_set} into others_RSM (no valid sessions)')
                                pass
                        others_RSM = np.mean(other_RSMs, axis=0)
                        others_RSMupper = get_upper_triangle(others_RSM)
                        # corr
                        corr = np.min(np.corrcoef(subject_RSMupper, others_RSMupper))
                        info2corr.append({'subject': subject, 'subject_maze': subject_maze, 'other_maze': others_maze,
                                          'goal_set': goal_set,'correlation': corr})
                    except IndexError: #no valid sessions for this (initial) subject, maze, goal_set combination
                        print(f'Failed assess {subject}, {subject_maze}, {goal_set} (no valid sessions)')
                        pass
    return pd.DataFrame(info2corr) 
                        


# %% Within Subject RSM Analysis Subfunctions 
    
def _get_subject_RSM(valid_sessions_df, decoding_kwargs, subject, maze_no, goal_set):
    # set up masks
    subject_mask = valid_sessions_df['subject_ID']==subject
    maze_mask = valid_sessions_df['maze']==maze_no+1
    goal_set_mask = valid_sessions_df['goal_set']==goal_set
    #
    subject_activity_df = valid_sessions_df[subject_mask & maze_mask & goal_set_mask]['activity_df'].to_list()[0]           
    subject_RSM = get_goal_representation_similarity_matrix(subject_activity_df)
    return subject_RSM

# %% Plot average RSM for each maze
goal_set_mask = valid_sessions_df['goal_set'] == 'subset_1'
for m in [1,2,3]:
    maze_mask = valid_sessions_df['maze']==m
    subject_RSMs = []
    for subject in exp_info['subject_IDs']:
        if subject != 'm3':
            subject_mask = valid_sessions_df['subject_ID'] == subject
            subject_activity_df = valid_sessions_df[subject_mask & maze_mask & goal_set_mask]['activity_df'].to_list()[0]
            subject_RSM = get_goal_representation_similarity_matrix(subject_activity_df)
            subject_RSMs.append(subject_RSM)
    plt.matshow(np.mean(subject_RSMs, axis=0))



# %% plotting within maze and across maze correlations

corr_within_mazes_df = correlate_subject_GRS_within_mazes(valid_sessions_df, exp_info)
corr_across_mazes_df = correlate_subject_GRS_across_mazes(valid_sessions_df, exp_info)
#stats 
def statistic(x, y):
    return np.mean(x) - np.mean(y)

from scipy.stats import permutation_test
res = permutation_test((corr_within_mazes_df.groupby(['subject']).mean(), corr_across_mazes_df.groupby(['subject']).mean()), statistic,
                       n_resamples=np.inf, alternative='two-sided')
# parametric t-test
# stat, pvalue = stats.wilcoxon(corr_within_mazes_df.groupby(['subject']).mean(), corr_across_mazes_df.groupby(['subject']).mean())

if 0.05>res.pvalue[0]>=0.01:
    stat = '*'
elif 0.01>res.pvalue[0]>=0.005:
    stat = '**'
elif res.pvalue[0]<0.005:
    stat = '***'
else:
    stat = 'ns'


corr_within_across_maze_df = pd.DataFrame()
corr_within_across_maze_df['same_maze'] = corr_within_mazes_df.groupby(['subject']).mean()
corr_within_across_maze_df['different_maze'] = corr_across_mazes_df.groupby(['subject']).mean()

f,ax = plt.subplots(1,1, figsize=(4,4))
sns.swarmplot(data = corr_within_across_maze_df, ax=ax, size=12, alpha=0.8)
ax.plot(corr_within_across_maze_df.T , color='grey')
ax.set_ylim(0, 0.5)
ax.set_ylabel('corr coef', fontweight='bold')
ax.set_xlabel('Goal RS Correlation', fontweight='bold')
ax.set_xticklabels(['within maze', 'across mazes'])
f.text(0.5, 0.85, f'{stat}', fontweight='bold', size=14)


# %% WIP
