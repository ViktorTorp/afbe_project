import numpy as np
import scipy.optimize as opt


class BerkFairLogreg:
    """
    Berk et al. (2017) model with fair constraint.
    Code is inspired from class exercise session 6.
    """

    def __init__(self, lambda_, gamma_=0.001, maxfun=500):
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.maxfun = maxfun
        self.model_trained = False

    def sigmoid(self, logits):
        """
        Calculate the sigmoid value from logits
        s_value = 1/(1+exp(-logits))
        """
        s_value = 1 + np.exp(-1 * logits)
        return 1/s_value

    def compute_cost(self, logits, X, y, lambda_, gamma_):
        '''computes cost function with constraints

        '''

        # number of training samples
        m = X.shape[0]

        # calculate the standard log loss function
        log_loss = -(1/m) * (y.dot(np.log(self.sigmoid(X.dot(self.weights)))
                                   ) + (1-y).dot(np.log(1-self.sigmoid(X.dot(self.weights)))))

        # calculate the group fairness loss
        # find the column for gender
        # Ensure that the protected col is places as the last column, i.e. col index -1
        S = X[:, -1]
        # calculate number of classes
        n1 = np.sum(S)
        n2 = len(S) - n1

        # calculate the group fairness cost
        cost_ = 0
        # we have to sum over all cross pairs
        for i in range(len(y)):
            # print(i)
            for j in range(i+1, len(y)):
                # check if labels are the same, distance is 1 if y[i] == y[j] and 0 if y[i] != y[j]
                if y[i] == y[j]:
                    # only calculate fairness for cross pairs
                    if X[i, -1] != X[j, -1]:  # in this case column 35 is the gender column
                        cost_ += (X[i].dot(self.weights)) - \
                            (X[j].dot(self.weights))

        # the total fairness cost is
        fair_cost = (cost_/(n1*n2))**2

        # put everything together
        J = log_loss + lambda_*fair_cost + \
            gamma_*np.linalg.norm(self.weights, 2)
        #J = log_loss + gamma_*np.linalg.norm(weights,2)

        return J

    def compute_gradient(self, weights, X, y, lambda_, gamma_):
        ''' calculate the gradient - used for finding the best weights values'''
        # empty gradient
        grad = np.zeros(weights.shape)

        m = len(X)  # number of training samples

        # calculate the sigmoid function
        h = self.sigmoid(X.dot(weights))

        # calculate gradients for each weights value
        for i in range(len(grad)):
            if i == 0:  # we do not want to regularize the intercept
                grad[i] = (1/m) * (h-y).dot(X[:, i])
            else:
                grad[i] = (1/m) * (h-y).dot(X[:, i]) + ((2*gamma_)*weights[i])

        return grad

    def fit(self, X_train, y_train):

        self.num_datapoints, self.num_features = X_train.shape
        self.weights = np.zeros(self.num_features)
        # solve to results for the lambda value
        result = opt.fmin_tnc(func=self.compute_cost, x0=self.weights, fprime=self.compute_gradient, maxfun=self.maxfun,
                              args=(X_train, y_train, self.lambda_, self.gamma_), xtol=1e-7)
        # the minimization function returns 3 things, 1) the found weights values, 2) no iteratations it took,
        # 3) the end state, 0,1,2 states are good - means it converged - other values are problematic
        self.weights, self.niter, self.out_state = result
        out_state_readable = "good" if self.out_state in [0, 1, 2] else "bad"
        print(
            f"The model end state is {self.out_state} which is {out_state_readable}")
        # to understand if our solver converged we print out the end state
        # print(lambda_,out_state)
        self.model_trained = True
        return self

    def predict(self, X_test):
        return X_test.dot(self.weights)
