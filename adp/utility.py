import warnings
import copy
import torch
import numpy as np
import scipy.optimize
import scipy.interpolate
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.metrics import check_scoring, get_scorer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, column_or_1d, check_X_y
from sklearn.kernel_ridge import KernelRidge

def _get_s_vec(curve, n_grid):
    try:
        only_categorical = curve.is_only_categorical()
    except AttributeError:
        only_categorical = False
    if only_categorical:
        return np.array([0, 1])  # Only return one value of s since only categorical
    return np.linspace(0, 1, num=n_grid)


class Utility():
    def __call__(self, curve, n_grid=50):
        # Take the average utility over all categorical settings
        s = _get_s_vec(curve, n_grid)
        try:
            args_list = [(s, z) for z in curve.get_possible_z_idx()]
        except AttributeError:
            args_list = [(s,)]  # Do not need to handle categorical
        return np.mean([self._single_curve(curve(*args), s) 
                        for args in args_list])


class ModelContrastUtility(Utility):
    def __init__(self, model, other_model, scoring='neg_mean_squared_error'):
        self.model = model
        self.other_model = other_model
        self.scoring = scoring

    def _single_curve(self, X_curve, s):
        scorer = get_scorer(self.scoring)
        y_curve = self.model(X_curve)
        class _WrapperEstimator():
            def predict(_, X):
                return self.other_model(X)
        score = scorer(_WrapperEstimator(), X_curve, y_curve)
        loss = -score
        return loss

    def get_comparison_model(self, x0):
        return self.other_model


class LeastConstantUtility(ModelContrastUtility):
    def __init__(self, model, scoring='neg_mean_squared_error'):
        self.model = model
        self.scoring = scoring

    def __call__(self, curve, n_grid=50):
        # Copied from Utility since need to include x0 
        # Take the average utility over all categorical settings
        s = _get_s_vec(curve, n_grid)
        try:
            args_list = [(s, z) for z in curve.get_possible_z_idx()]
        except AttributeError:
            args_list = [(s,)]  # Do not need to handle categorical
        return np.mean([self._single_curve(curve(*args), s, curve.x0) 
                        for args in args_list])

    def _single_curve(self, X_curve, s, x0):
        scorer = get_scorer(self.scoring)
        y_curve = self.model(X_curve)
        other_model = self.get_comparison_model(x0)
        class _WrapperEstimator():
            def predict(_, X):
                return other_model(X)
        score = scorer(_WrapperEstimator(), X_curve, y_curve)
        loss = -score
        return self._multiplier() * loss

    def _multiplier(self):
        return 1

    def get_comparison_model(self, x0):
        y_constant = self.model(x0.reshape(1, -1))[0]
        def _constant_model(X):
            X = check_array(X)
            return y_constant * np.ones(X.shape[0])
        return _constant_model


class MostConstantUtility(LeastConstantUtility):
    def _multiplier(self):
        return -1


class BestPartialModelUtility(Utility):
    def __init__(self, model, regression_estimator, scoring='neg_mean_squared_error'):
        self.model = model
        self.regression_estimator = regression_estimator
        self.scoring = scoring

    def _single_curve(self, X_curve, s):
        scorer = check_scoring(self.regression_estimator, scoring=self.scoring)
        s = np.array(s)

        # Fit regression model
        y_curve = self.model(X_curve)
        reg = clone(self.regression_estimator)
        reg.fit(s.reshape(-1, 1), y_curve)
        
        # Score function
        score = scorer(reg, s.reshape(-1,1), y_curve)
        loss = -score
        
        self.regressor_ = reg
        self.scorer_ = scorer
        return loss

    def fit_model(self, X, y):
        reg = clone(self.regression_estimator)
        reg.fit(X, y)
        return reg


class LeastLinearUtility(BestPartialModelUtility):
    def __init__(self, model, scoring='neg_mean_squared_error'):
        self.model = model
        self.scoring = scoring
        self.regression_estimator = LinearRegression()


class LeastKernelRidgeUtility(BestPartialModelUtility):
    def __init__(self, model, scoring='neg_mean_squared_error', kernel='rbf', gamma=0.001, alpha=1e-13, **kernel_ridge_params):
        self.model = model
        self.scoring = scoring
        self.regression_estimator = KernelRidge(kernel=kernel, gamma=gamma, alpha=alpha, **kernel_ridge_params)


class FixedIsotonicRegression(IsotonicRegression):
    """
    This class fixes the issue that IsotonicRegression takes
    only a vector x.

    In addition, we make auto actually try both increasing
    and decreasing rather than determining via Spearman's
    correlation when increasing = 'auto'.
    """

    def fit(self, X, y=None):
        X = self._check_X(X)
        if self.increasing == 'auto':
            self.increasing = True
            super(FixedIsotonicRegression, self).fit(X, y)
            score_inc = self.score(X, y)

            self.increasing = False
            super(FixedIsotonicRegression, self).fit(X, y)
            score_dec = self.score(X, y)
            
            if score_inc > score_dec:
                # Refit because increasing was better
                self.increasing = True
                super(FixedIsotonicRegression, self).fit(X, y)
            
            # Reset increasing parameter
            self.increasing = 'auto'
        return self
    
    def predict(self, X):
        X = self._check_X(X)
        return super(FixedIsotonicRegression, self).predict(X)
        
    def _check_X(self, X):
        X = column_or_1d(X)  # Convert single column or 1d vector to 1d vector
        return X


class LeastMonotonicUtility(BestPartialModelUtility):
    def __init__(self, model, scoring='neg_mean_squared_error'):
        self.model = model
        self.scoring = scoring
        self.regression_estimator = FixedIsotonicRegression(
            increasing='auto', out_of_bounds='clip')


# Alias for backwards compatibility
MonotonicUtility = LeastMonotonicUtility


class _LipschitzRegressor(BaseEstimator, RegressorMixin):
    """
    This class implements Lipschitz-bounded regression via 
    least squares.
    """
    def __init__(self, lipschitz_constant):
        self.lipschitz_constant = lipschitz_constant

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        x = self._check_X(X)
        n_samples = x.shape[0]
        L = self.lipschitz_constant

        # Construct optimization problem and solve
        A = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            if i == 0:
                A[:,i] = 1
            else:
                A[i:,i] = (x[i] - x[i-1])
        b = y.copy()
        lower = -L * np.ones(n_samples)
        lower[0] = -np.inf
        upper = L * np.ones(n_samples)
        upper[0] = np.inf
        bounds = (lower, upper)

        # Compute optimization
        result = scipy.optimize.lsq_linear(A=A, b=b, bounds=bounds)
        x_opt = result.x
        yhat = np.dot(A, x_opt)

        self.x_vals_ = x
        self.y_vals_ = yhat
        return self
    
    def predict(self, X):
        x = self._check_X(X)
        interp = scipy.interpolate.interp1d(
            self.x_vals_, self.y_vals_, kind='linear', copy=False, fill_value='extrapolate')
        ypred = interp(x)
        return ypred
        
    def _check_X(self, X):
        X = column_or_1d(X)  # Convert single column or 1d vector to 1d vector
        return X


class LeastLipschitzUtility(BestPartialModelUtility):
    def __init__(self, model, lipschitz_constant=1, scoring='neg_mean_squared_error'):
        self.model = model
        self.scoring = scoring
        self.regression_estimator = _LipschitzRegressor(lipschitz_constant)


class TotalVariationUtility(Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s):
        if len(s) < 2:
            return 0
        bandwidth = s[1] - s[0]
        f_out = self.model(X_curve)
        grad_f_out = np.gradient(f_out, s)
        return np.mean(np.abs(grad_f_out))


class MostJerkUtility(Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s):
        if len(s) < 2:
            return 0
        bandwidth = s[1] - s[0]
        f_out = self.model(X_curve)
        grad_f_out = np.gradient(f_out, s)
        grad2_f_out = np.gradient(grad_f_out, s)
        grad3_f_out = np.gradient(grad2_f_out, s)
        return self._multiplier() * np.mean(np.abs(grad3_f_out))

    def _multiplier(self):
        return 1.0


class LeastJerkUtility(MostJerkUtility):
    def _multiplier(self):
        return -1.0


class MostCurvatureUtility(Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s):
        if len(s) < 2:
            return 0
        bandwidth = s[1] - s[0]
        f_out = self.model(X_curve)
        grad_f_out = np.gradient(f_out, s)
        grad2_f_out = np.gradient(grad_f_out, s)
        return self._multiplier() * np.mean(np.abs(grad2_f_out))

    def _multiplier(self):
        return 1.0


class LeastCurvatureUtility(MostCurvatureUtility):
    def _multiplier(self):
        return -1.0
    

class ConditionalLinearUtility(Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s):
        f_out = self.model(X_curve)
        linear_model = self.fit_linear_model(s, f_out)
        g_out = linear_model(s)
        return np.mean(np.abs(f_out - g_out))

    def fit_linear_model(self, x, y):
        t_new = np.array([np.ones(np.asarray(x).shape[0]), x]).transpose()
        beta_star = np.dot(np.dot(np.linalg.inv(np.dot(t_new.T, t_new)), t_new.T), y)
        def linear_model(t):
            return t * beta_star[1] + beta_star[0]
        return linear_model


class UniquenessUtility(Utility):
    def __init__(self, model, X_sample):
        self.model = model
        self.X_sample = X_sample
        
    def __call__(self, curve, n_grid=50):

        # Setup s and categorical args_list
        s = _get_s_vec(curve, n_grid)
        try:
            args_list = [(s, z) for z in curve.get_possible_z_idx()]
        except AttributeError:
            args_list = [(s,)]  # Do not need to handle categorical

        # Get other curves
        if len(self.X_sample) > 30:
            warnings.warn('More than 30 samples in X_sample, computing curves for all samples, might be slow')
        other_curves = self.get_other_curves(curve)

        # Mean over f_out, samples, categorical values
        results = np.array([
            self._curve_difference(curve, other_curve, args_list)
            for other_curve in other_curves
        ])
        return np.mean(results)

    def get_other_curves(self, curve):
        def new_curve(x0):
            new_curve = copy.deepcopy(curve)
            new_curve.x0 = x0
            # Fix Z_list to be consistent with x0
            if hasattr(new_curve, 'Z_list') and new_curve.Z_list is not None:
                #print('x0', new_curve.x0)
                sel_changing = curve._is_changing_categorical()
                def fix_z(z):
                    z = np.array(z).copy()
                    z[~sel_changing] = x0[new_curve.is_categorical()][~sel_changing]
                    return z
                new_curve.Z_list = [fix_z(z) for z in new_curve.Z_list]
            return new_curve
        return [new_curve(x) for x in self.X_sample]

    def mean_shifted_val_list(self, other_curve, args_list, mean_f):
        g_out_list, mean_g = self._curve_vals_and_mean(other_curve, args_list)
        return [g_out - mean_g + mean_f for g_out in g_out_list]

    def _curve_vals_and_mean(self, curve, args_list):
        vals_list = [
            np.array(self.model(curve(*args)))
            for args in args_list
        ]
        # Note: Technically vals could be different lengths depending on args_list
        mean = np.mean([np.mean(vals) for vals in vals_list]) 
        return vals_list, mean

    def _curve_difference(self, curve, other_curve, args_list):
        f_out_list, mean_f = self._curve_vals_and_mean(curve, args_list)
        g_shifted_out_list = self.mean_shifted_val_list(other_curve, args_list, mean_f)
        return  np.mean([
            np.mean(np.abs(f_out - g_shifted_out))
            for f_out, g_shifted_out in zip(f_out_list, g_shifted_out_list)
        ])


class GradientUniquenessUtility(GeneralizabilityUtility):
    def _curve_difference(self, curve, other_curve, args_list):
        def get_gradients(curve, args):
            X_curve = curve(*args)
            f_out = self.model(X_curve)
            grad_f_out = np.diff(f_out)
            return grad_f_out
        return np.mean([
            np.mean(np.abs(get_gradients(curve, args) - get_gradients(other_curve, args)))
            for args in args_list
        ])
