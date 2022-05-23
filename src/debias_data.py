import numpy as np
import scipy


def reproject_features(data, protected_cols, nonprotected_cols):
    """
    generate a fair representation of nonprotected columns which are independent from any columns in protected_cols
    data: a data frame
    protected_cols: list of strings, the protected columns
    nonprotected_col: string, all other data columns 

    NOTE: this function assumes the data is already centered 
    """
    # make a copy of data
    df = data.copy()
    # Protected features
    protect = df[protected_cols].values
    # extract data about nonprotected columns
    debiased_nonprotect = df[nonprotected_cols].values
    # crease an orthonormal basis
    base_protect = scipy.linalg.orth(protect)
    for j in range(debiased_nonprotect.shape[1]):
        debiased_nonprotect[:,
                            j] -= base_protect @ base_protect.T @ debiased_nonprotect[:, j]
    return debiased_nonprotect


def reproject_features_w_regul(data, protected_cols, nonprotected_cols, lambda_):
    """
    generate a fair representation of nonprotected columns which are independent from any columns in protected_cols
    dat_: a data frame
    protected_cols: list of strings, the protected columns
    nonprotected_col: string, all other data columns 
    lambda_: float number between 0 and 1, 0 means totally fair; 1 means same as raw data
    """

    # run the normal reproject_features function
    r = reproject_features(data, protected_cols, nonprotected_cols)

    # extract data about nonprotected variables
    nonprotect = data[nonprotected_cols].values
    # standardize columns

    return r + lambda_*(nonprotect - r)


def debais_data(protected_cols, nonprotected_cols, lambda_=0):
    """
    Debias data, data pipeline

    Arguments
    ---------
    protected_cols: list of strings, 
        the protected columns

    nonprotected_col: String, 
        all other data columns 

    Returns
    -------
    Function(X_train, X_test) => X_train_r, X_test_r
        A fucntion which takes X_train, X_test and returns the reprojected versions of X_train, X_test

    """
    def call(X_train, X_test):
        # Standard scale the data data
        _mean = np.mean(X_train, axis=0)
        _std = np.std(X_train, ddof=1, axis=0)
        X_train_scaled = (X_train - _mean)/_std
        X_test_scaled = (X_test - _mean)/_std
        # reproject features
        X_train_r = reproject_features_w_regul(
            X_train_scaled, protected_cols, nonprotected_cols, lambda_)
        X_test_r = reproject_features_w_regul(
            X_test_scaled, protected_cols, nonprotected_cols, lambda_)
        return X_train_r, X_test_r
    return call
