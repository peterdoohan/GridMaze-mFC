# %% Imports
from GridMaze.analysis.neGLM import get_input_data as gid
from GridMaze.analysis.neGLM import run_neGLM as rne
from neGLM.models import neGLM
from importlib import reload


# %%

# get input data
# input_data = get_input_data(input_groups=["distance_to_goal", "place_direction"])

# # divide training data
# i = 3
# test_session = input_data[i]  # single session
# train_sessions = input_data[:i] + input_data[i + 1 :]  # all other sessions

# # init model
# model = neGLM(Nhid=[100, 50], Nlat=10, partition=None, latent_split=None)

# # train
# model.train(train_sessions, test_session, nepochs=5001, n_jobs=-1, verbose=True)

# # test
# test_perf = model.score(
#     x=test_session["X"],
#     y=test_session["spikes"],
#     trials=test_session["trial_ids"],
#     n_folds=5,
#     optimal_alpha=True,
#     n_jobs=-1,
#     verbose=False,
# )
# cluster_cv_scores = rne._get_cluster_cross_val_df(
#     test_perf,
#     test_session["session_info"],
#     test_session["cluster_unique_IDs"],
# )

# %% test a few different models


def model_comparison_test():
    i = 3
    mean_perfs = []
    input_features = []
    for additional_feature in ["place_direction", "distance_to_goal", "goal", "egocentric_action", "velocity"]:
        input_features.append(additional_feature)
        print(input_features)
        input_data = gid.get_input_data(input_groups=input_features)
        test_session = input_data[i]  # single session
        train_sessions = input_data[:i] + input_data[i + 1 :]  # all other sessions
        model = neGLM(Nhid=[100, 50], Nlat=10, partition=None, latent_split=None)
        model.train(train_sessions, test_session, nepochs=5001, n_jobs=-1, verbose=True)
        test_perf = model.score(
            x=test_session["X"],
            y=test_session["spikes"],
            trials=test_session["trial_ids"],
            n_folds=5,
            optimal_alpha=True,
            n_jobs=-1,
            verbose=False,
        )
        print(test_perf.mean())
        mean_perfs.append(test_perf.mean())
    return mean_perfs
