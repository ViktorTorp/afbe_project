import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import scipy.cluster.hierarchy as sch

# Helping function to cluster correlations from https://wil.yegelwel.com/cluster-correlation-matrix/


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def spearmanr_pval(x, y):
    return spearmanr(x, y)[1]


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


def get_significant_correlations(df, method="pearson", p_val=0.05, cluster=True):
    corr = df.corr(method)
    # Bonferroni correction
    corrected_p_val = p_val / (len(df.columns)**2/2)

    if method == "spearman":
        corr_pvals = df.corr(spearmanr_pval)
    elif method == "pearson":
        corr_pvals = df.corr(pearsonr_pval)

    # Set significant values to 1
    significant_matrix = np.ones(corr.shape)
    significant_matrix[corr_pvals >= corrected_p_val] = 0
    # Set un-significant correlations to 0
    corr = corr * significant_matrix

    # Return clusrtered corr
    if cluster:
        return cluster_corr(corr)
    else:
        return corr


#fig, ax = plt.subplots(1, figsize=(15,15))
#sns.heatmap(get_significant_correlations(df_init), center=0, annot=True, cmap="viridis")
