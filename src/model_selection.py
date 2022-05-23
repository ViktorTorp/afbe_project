import numpy as np
import pandas as pd


def test_model(model, X, y, kfold, scaler=None, model_params={}):
    """
    Test models using some cross validation method and with a scaling method.

    Parameters
    ----------
    model : sklearn model object. 
        This object must not be instansiated

    X : pandas.Dataframe
        df containing all the features, which the model should be trained on

    y : np.array
        An array on the taget values

    kfold : sklearn.model_selection method e.g. KFold
        This object must be initialised. See example

    scaler : Function, default=None
        Data processing pipeline of the form 
        (X_train, X_test) => X_train_processed, X_test_processed
        If None, then are the data not processed

    model_params : Dict, default=Dict
        A dictionary containing model parameters


    Returns
    -------
    numpy.array
        An array containing the predictions


    Example
    -------
    X, y = df.drop(columns=['credit_risk']), df['credit_risk'].values
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    process_data = scale_data()
    model = LogisticRegression
    preds = test_model(model, X, y, kfold, process_data)
    """

    preds = np.zeros(len(y))

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = np.array(y)[train_index]
        if scaler:
            X_train, X_test = scaler(X_train, X_test)

        tmp_model = model(**model_params)
        tmp_model = tmp_model.fit(X_train, y_train)
        preds[test_index] = tmp_model.predict(X_test)
    return preds
