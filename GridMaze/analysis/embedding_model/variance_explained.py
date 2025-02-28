"""
Library for calculating the (pseudo) variance explained by different input features from embedding model regression results.
Eg, full model vs reduced mode.
"""

# %% Imports
import pandas as pd
from GridMaze.analysis.embedding_model.input_feature_comparisons import _load_results
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from matplotlib_venn import venn3
from matplotlib import pyplot as plt


# %% Global variables
EXP_SET = "var_explained2"

# %% Functions


def get_variance_explained_results(subject_ID="m2", bonferroni_correction=False, alpha=0.05):
    """ """
    results_df = _load_results(EXP_SET, average_over_folds=False)
    # look at one subject at a time for now
    if subject_ID != "all":
        results_df = results_df[results_df["subject_ID"] == subject_ID]
    results_pivot = results_df.set_index(["cluster_unique_ID", "fold"])[["cv_performance", "abbrev"]].pivot(
        columns="abbrev", values="cv_performance"
    )  # index: cluster, columns: cv perf under full & reduced models
    # get mean var exp across folds
    features = ["distance", "goal", "place_direction", "trial_phase"]
    mean_model_perfs = results_pivot.groupby("cluster_unique_ID").mean()
    var_exp_df = pd.DataFrame(
        index=mean_model_perfs.index, columns=["distance", "goal", "place_direction", "trial_phase"]
    )
    for feature in features:
        var_exp_df[feature] = mean_model_perfs["full_model"] - mean_model_perfs[f"reduced_{feature}"]
    # get p-values for each cluster-feature (t-test, see get_cluster_p_values)
    cluster_unique_IDs = results_pivot.index.get_level_values(0).unique()
    p_values_df = pd.DataFrame(index=cluster_unique_IDs, columns=features)
    for cluster in cluster_unique_IDs:
        feature2p_value = get_cluster_p_values(results_pivot.loc[cluster])
        p_values_df.loc[cluster] = list(feature2p_value.values())
    if bonferroni_correction:
        for feature in features:
            p_values_df[feature] = multipletests(
                p_values_df[feature].values,
            )[1]
    sig_df = p_values_df.lt(alpha)
    plot_venn(sig_df)
    return var_exp_df, sig_df


def get_cluster_p_values(cluster_results):
    """ """
    mean_full_model_perf = cluster_results.full_model.mean()
    feature2p_value = {}
    for reduced_mod in ["reduced_distance", "reduced_goal", "reduced_place_direction", "reduced_trial_phase"]:
        reduced_perf = (mean_full_model_perf - cluster_results[reduced_mod]).values
        _, p_value = ttest_1samp(reduced_perf, 0, alternative="greater")
        feature2p_value[reduced_mod] = p_value
    return feature2p_value


# %% plotting


def plot_venn(df, ax=None):
    """"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    venn_counts = {
        "100": len(df[(df["distance"]) & (~df["place_direction"]) & (~df["trial_phase"])]),
        "010": len(df[(~df["distance"]) & (df["place_direction"]) & (~df["trial_phase"])]),
        "001": len(df[(~df["distance"]) & (~df["place_direction"]) & (df["trial_phase"])]),
        "110": len(df[(df["distance"]) & (df["place_direction"]) & (~df["trial_phase"])]),
        "101": len(df[(df["distance"]) & (~df["place_direction"]) & (df["trial_phase"])]),
        "011": len(df[(~df["distance"]) & (df["place_direction"]) & (df["trial_phase"])]),
        "111": len(df[(df["distance"]) & (df["place_direction"]) & (df["trial_phase"])]),
    }

    # Create the Venn diagram for 'distance', 'place_direction', and 'trial_phase'
    venn = venn3(
        subsets=(
            venn_counts["100"],
            venn_counts["010"],
            venn_counts["110"],
            venn_counts["001"],
            venn_counts["101"],
            venn_counts["011"],
            venn_counts["111"],
        ),
        set_labels=("distance", "place_direction", "trial_phase"),
        ax=ax,
    )
