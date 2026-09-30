"""
from-scratch multinomial logistic regression for AT82.03 A3

Based on the LogisticRegression class from "02 - Multinomial Logistic Regression.ipynb".

  Task 1 (metrics from scratch)
      converting label selling price into discrete variable using bins
      confusion_matrix, support
      accuracy
      precision / recall / f1_score            -> one score per class
      macro_precision / macro_recall / macro_f1        -> plain average over classes
      weighted_precision / weighted_recall / weighted_f1 -> average weighted by support
      classification_report                  

  Task 2 (ridge)
      NoPenalty, RidgePenalty                 
      RidgeLogisticRegression                 

Usage:
    from logistic_regression import LogisticRegression, RidgeLogisticRegression

    model = LogisticRegression(k=4, n=X_train.shape[1], method='minibatch', alpha=0.01)
    model.fit(X_train, Y_train_onehot)
    y_pred = model.predict(X_test)
    print(model.classification_report(y_test, y_pred))

"""

import time

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# A3 Task 2: penalties.
# ===========================================================================
class NoPenalty:
    """No regularization -> plain logistic regression."""

    def __init__(self, l=0.0):
        self.l = 0.0

    def __call__(self, W):
        return 0.0

    def derivation(self, W):
        return np.zeros_like(W)


class RidgePenalty:
    """L2 (ridge) penalty:  lambda * sum(W^2),  gradient  2 * lambda * W."""

    def __init__(self, l):
        self.l = l

    def __call__(self, W):
        return self.l * np.sum(np.square(W))

    def derivation(self, W):
        return self.l * 2 * W


class LogisticRegression:
    """Multinomial logistic regression trained with gradient descent.

    Parameters
    ----------
    k : int          number of classes
    n : int          number of features INCLUDING the intercept column
    method : str     'batch', 'minibatch' or 'sto'
    alpha : float    learning rate
    max_iter : int   number of gradient-descent iterations
    regularization : A3 - penalty object (NoPenalty or RidgePenalty). 
                    None -> NoPenalty.
                    init : str       A3 - 'zeros' or 'random' weight initialisation
    verbose : bool   A3 - print the loss while training
    """

    def __init__(self, k, n, method='batch', alpha=0.001, max_iter=5000,
                 regularization=None, init='zeros', verbose=True, seed=None):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        # A3: default to no penalty so the class still behaves like the original one
        self.regularization = NoPenalty() if regularization is None else regularization
        self.init = init
        self.verbose = verbose
        self.seed = seed
        if method not in ('batch', 'minibatch', 'sto'):
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

    # ---------------------------------------------------------------- weights
    def _init_weights(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.init == 'zeros':
            self.W = np.zeros((self.n, self.k))
        elif self.init == 'random':
            self.W = np.random.rand(self.n, self.k)
        else:
            raise ValueError("init must be 'zeros' or 'random'")

    # -------------------------------------------------------------------- fit
    def fit(self, X, Y):
        """X: (m, n) with intercept column. Y: (m, k) one-hot."""
        self._init_weights()
        self.losses = []

        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if self.verbose and i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            if self.verbose:
                print(f"time taken: {time.time() - start_time}")

        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])  # with replacement
                batch_X = X[ix:ix + batch_size]
                batch_Y = Y[ix:ix + batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if self.verbose and i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            if self.verbose:
                print(f"time taken: {time.time() - start_time}")

        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while idx in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx].reshape(1, -1)  
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad

                list_of_used_ix.append(idx)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if self.verbose and i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            if self.verbose:
                print(f"time taken: {time.time() - start_time}")

        return self

    # --------------------------------------------------------------- gradient
    def gradient(self, X, Y):
        """Cross-entropy loss and its gradient, both averaged over the m samples
        in this batch, plus the A3 penalty term."""
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        # clip before the log so a probability of exactly 0 cannot give -inf
        loss = - np.sum(Y * np.log(np.clip(h, 1e-15, 1.0))) / m
        error = h - Y
        grad = self.softmax_grad(X, error)

        # add the penalty. The intercept row W[0] is left out on purpose:
        # it only sets the base rate of each class and shrinking it towards zero
        # would just bias every prediction, which is not what regularization is for.
        penalty_W = self.W.copy()
        penalty_W[0] = 0.0
        loss = loss + self.regularization(penalty_W)
        grad = grad + self.regularization.derivation(penalty_W)
        return loss, grad

    def softmax(self, theta_t_x):
        # subtract the row max before exp. This changes nothing mathematically
        # (the constant cancels in the ratio) but stops np.exp from overflowing
        # when X @ W gets large, which happened with a high learning rate.
        z = theta_t_x - np.max(theta_t_x, axis=1, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        #  divided by m. The original returned X.T @ error while the loss was
        # divided by m, so loss and gradient were on different scales. That is
        # harmless on its own, but once a penalty is added the two terms have to
        # be comparable, otherwise lambda does not mean what it should.
        return X.T @ error / X.shape[0]

    def h_theta(self, X, W):
        """
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        """
        return self.softmax(X @ W)

    # ---------------------------------------------------------------- predict
    def predict(self, X_test):
        """Predicted class label (0..k-1) for each row."""
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def predict_proba(self, X_test):
        """A3: class probabilities, shape (m, k). Useful for the web app."""
        return self.h_theta(X_test, self.W)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    # =======================================================================
    # A3 Task 1: classification metrics, all built from one confusion matrix.
    # =======================================================================
    def confusion_matrix(self, y_true, y_pred):
        """(k, k) matrix. Rows = actual class, columns = predicted class, so
        cm[i, j] counts samples of class i that were predicted as class j."""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        cm = np.zeros((self.k, self.k), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    def _tp_fp_fn(self, y_true, y_pred):
        """Per-class counts read off the confusion matrix.
        For class c:
            TP = cm[c, c]                    -> actually c, predicted c
            FP = column c minus TP           -> predicted c, actually something else
            FN = row c minus TP              -> actually c, predicted something else
        """
        cm = self.confusion_matrix(y_true, y_pred)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp     # down the column
        fn = cm.sum(axis=1) - tp     # across the row
        return tp, fp, fn

    @staticmethod
    def _safe_divide(numerator, denominator):
        """0 / 0 = 0 instead of nan. sklearn does the same thing with its
        zero_division=0 option: a class that was never predicted gets a
        precision of 0 rather than an undefined value."""
        numerator, denominator = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
        out = np.zeros_like(numerator, dtype=float)
        nonzero = denominator != 0
        out[nonzero] = numerator[nonzero] / denominator[nonzero]
        return out

    def support(self, y_true):
        """A3: number of samples that truly belong to each class.
        This is the "support" column in sklearn's classification report and it is
        also the weighting factor used by the weighted averages below."""
        y_true = np.asarray(y_true).astype(int)
        return np.array([(y_true == c).sum() for c in range(self.k)])

    # ------------------------------------------------------------- accuracy
    def accuracy(self, y_true, y_pred):
        """correct predictions / all predictions -> a single number."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.sum(y_true == y_pred) / len(y_true))

    # ------------------------------------------------- per-class metrics
    def precision(self, y_true, y_pred):
        """precision_c = TP_c / (TP_c + FP_c). Of everything we labelled class c,
        how much really was class c. Returns one value per class."""
        tp, fp, _ = self._tp_fp_fn(y_true, y_pred)
        return self._safe_divide(tp, tp + fp)

    def recall(self, y_true, y_pred):
        """recall_c = TP_c / (TP_c + FN_c). Of all the real class-c samples,
        how many we managed to find. Returns one value per class."""
        tp, _, fn = self._tp_fp_fn(y_true, y_pred)
        return self._safe_divide(tp, tp + fn)

    def f1_score(self, y_true, y_pred):
        """f1_c = 2 * precision_c * recall_c / (precision_c + recall_c),
        the harmonic mean, so a class only scores well if BOTH are good."""
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return self._safe_divide(2 * p * r, p + r)

    # ------------------------------------------------------ macro averages
    # Plain mean over the classes. Every class counts the same, however rare it is.
    def macro_precision(self, y_true, y_pred):
        return float(np.mean(self.precision(y_true, y_pred)))

    def macro_recall(self, y_true, y_pred):
        return float(np.mean(self.recall(y_true, y_pred)))

    def macro_f1(self, y_true, y_pred):
        return float(np.mean(self.f1_score(y_true, y_pred)))

    # --------------------------------------------------- weighted averages
    # Same averages, but each class is weighted by its share of the samples
    # (support / total). Big classes pull the score towards their own value,
    # which is what you want when the classes are imbalanced.
    def _weights(self, y_true):
        sup = self.support(y_true)
        return sup / sup.sum()

    def weighted_precision(self, y_true, y_pred):
        return float(np.sum(self._weights(y_true) * self.precision(y_true, y_pred)))

    def weighted_recall(self, y_true, y_pred):
        return float(np.sum(self._weights(y_true) * self.recall(y_true, y_pred)))

    def weighted_f1(self, y_true, y_pred):
        return float(np.sum(self._weights(y_true) * self.f1_score(y_true, y_pred)))

    # ------------------------------------------------------------- report
    def classification_report(self, y_true, y_pred, target_names=None, digits=2):
        """A3: prints the same table as sklearn.metrics.classification_report,
        built entirely from the functions above."""
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        f = self.f1_score(y_true, y_pred)
        sup = self.support(y_true)
        if target_names is None:
            target_names = [str(c) for c in range(self.k)]
        width = max(12, max(len(name) for name in target_names) + 2)

        lines = [f"{'':>{width}} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}", ""]
        for c in range(self.k):
            lines.append(f"{target_names[c]:>{width}} {p[c]:>9.{digits}f} {r[c]:>9.{digits}f} "
                         f"{f[c]:>9.{digits}f} {sup[c]:>9d}")
        lines.append("")
        acc = self.accuracy(y_true, y_pred)
        lines.append(f"{'accuracy':>{width}} {'':>9} {'':>9} {acc:>9.{digits}f} {sup.sum():>9d}")
        lines.append(f"{'macro avg':>{width}} {self.macro_precision(y_true, y_pred):>9.{digits}f} "
                     f"{self.macro_recall(y_true, y_pred):>9.{digits}f} "
                     f"{self.macro_f1(y_true, y_pred):>9.{digits}f} {sup.sum():>9d}")
        lines.append(f"{'weighted avg':>{width}} {self.weighted_precision(y_true, y_pred):>9.{digits}f} "
                     f"{self.weighted_recall(y_true, y_pred):>9.{digits}f} "
                     f"{self.weighted_f1(y_true, y_pred):>9.{digits}f} {sup.sum():>9d}")
        return "\n".join(lines)


# ===========================================================================
# A3 Task 2: convenience subclasses, same idea as RidgeRegression in A2.
# ===========================================================================
class NormalLogisticRegression(LogisticRegression):
    """Plain multinomial logistic regression (no penalty)."""

    def __init__(self, k, n, method='batch', alpha=0.001, max_iter=5000, **kwargs):
        super().__init__(k, n, method=method, alpha=alpha, max_iter=max_iter,
                         regularization=NoPenalty(), **kwargs)


class RidgeLogisticRegression(LogisticRegression):
    """Multinomial logistic regression with an L2 penalty:

        J(W) = -sum(y * log(h)) / m  +  lambda * sum(W^2)
    """

    def __init__(self, k, n, l=0.01, method='batch', alpha=0.001, max_iter=5000, **kwargs):
        super().__init__(k, n, method=method, alpha=alpha, max_iter=max_iter,
                         regularization=RidgePenalty(l), **kwargs)
        self.l = l


# ---------------------------------------------------------------------------
# helper used by the notebook and the web app
# ---------------------------------------------------------------------------
def one_hot(y, k):
    """(m,) integer labels -> (m, k) one-hot matrix, as the notebook does by hand."""
    y = np.asarray(y).astype(int)
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1
    return Y
