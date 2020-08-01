import warnings
import copy
import torch
import numpy as np
import scipy.optimize
import scipy.interpolate
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
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


class _Utility():
    def __call__(self, curve, n_grid=50):
        # Take the average utility over all categorical settings
        s = _get_s_vec(curve, n_grid)
        try:
            args_list = [(s, z) for z in curve.get_possible_z_idx()]
        except AttributeError:
            args_list = [(s,)]  # Do not need to handle categorical
        return np.mean([self._single_curve(curve(*args), s, curve.x0) 
                        for args in args_list])


##############################################################################
# Model contrast utilities
##############################################################################


class ModelContrastUtility(_Utility):
    def __init__(self, model, other_model, loss_metric=None):
        self.model = model
        self.other_model = other_model
        self.loss_metric = loss_metric

    def _single_curve(self, X_curve, s, x0):
        y_curve = self.model(X_curve)
        other_model = self.get_comparison_model(x0)
        y_other_curve = other_model(X_curve)
        return self.from_plot_vals(s, y_curve, y_other_curve, loss_metric=self.loss_metric)

    def get_comparison_model(self, x0):
        return self.other_model

    @classmethod
    def from_plot_vals(cls, x, y, y_other, loss_metric=None):
        if loss_metric is None:
            loss_metric = mean_squared_error
        assert np.allclose(np.diff(x), x[1]-x[0]), 'x should be evenly spaced points'
        return cls._multiplier() * loss_metric(y, y_other)

    @classmethod
    def _multiplier(cls):
        return 1


class LeastConstantUtility(ModelContrastUtility):
    def __init__(self, model, loss_metric=None):
        self.model = model
        self.loss_metric = loss_metric

    def get_comparison_model(self, x0):
        y_constant = self.model(x0.reshape(1, -1))[0]
        def _constant_model(X):
            X = check_array(X)
            return y_constant * np.ones(X.shape[0])
        return _constant_model


class MostConstantUtility(LeastConstantUtility):
    @classmethod
    def _multiplier(self):
        return -1


##############################################################################
# Functional property validation utilities
##############################################################################


class FunctionalPropertyUtility(_Utility):
    def __init__(self, model, regression_estimator=None, loss_metric=None):
        self.model = model
        self.regression_estimator = regression_estimator
        self.loss_metric = loss_metric

    def _single_curve(self, X_curve, s, x0):
        s = np.array(s)
        y_curve = self.model(X_curve)
        return FunctionalPropertyUtility.from_plot_vals(s, y_curve, regression_estimator=self.regression_estimator, loss_metric=self.loss_metric)

    def fit_model(self, X, y):
        reg = clone(self.regression_estimator)
        reg.fit(X, y)
        return reg

    @classmethod
    def from_plot_vals(cls, x, y, regression_estimator=None, loss_metric=None):
        if regression_estimator is None:
            regression_estimator = cls._default_regressor()
        if loss_metric is None:
            loss_metric = mean_squared_error
        assert np.allclose(np.diff(x), x[1]-x[0]), 'x should be evenly spaced points'

        # Fit regression model
        reg = clone(regression_estimator)
        reg.fit(x.reshape(-1, 1), y)

        # Predict and compute loss
        y_other = reg.predict(x.reshape(-1, 1))
        return cls._multiplier() * loss_metric(y, y_other)

    @classmethod
    def _default_regressor(cls, **kwargs):
        return LinearRegression(**kwargs)

    @classmethod
    def _multiplier(cls):
        return 1

class _ConcreteFunctionalPropertyUtility(FunctionalPropertyUtility):
    def __init__(self, model, regressor_kwargs=None, loss_metric=None):
        self.model = model
        self.loss_metric = loss_metric
        if regressor_kwargs is None:
            regressor_kwargs = {}
        self.regression_estimator = self._default_regressor(**regressor_kwargs)

    @classmethod
    def from_plot_vals(cls, x, y, regressor_kwargs=None, loss_metric=None):
        if regressor_kwargs is None:
            regressor_kwargs = {}
        if loss_metric is None:
            loss_metric = mean_squared_error
        assert np.allclose(np.diff(x), x[1]-x[0]), 'x should be evenly spaced points'

        reg = cls._default_regressor(**regressor_kwargs)

        return FunctionalPropertyUtility.from_plot_vals(
            x, y, regression_estimator=reg, loss_metric=loss_metric)


class LeastLinearUtility(_ConcreteFunctionalPropertyUtility):
    @classmethod
    def _default_regressor(cls, **kwargs):
        return LinearRegression(**kwargs)


class LeastKernelRidgeUtility(_ConcreteFunctionalPropertyUtility):
    @classmethod
    def _default_regressor(cls, **kwargs):
        both_kwargs = dict(kernel='rbf', gamma=0.001, alpha=1e-13)
        both_kwargs.update(kwargs)
        return KernelRidge(**both_kwargs)


class _FixedIsotonicRegression(IsotonicRegression):
    """
    This class fixes the issue that IsotonicRegression takes
    only a vector x.

    In addition, we make auto actually try both increasing
    and decreasing rather than determining via Spearman's
    correlation when increasing = 'auto'
    """

    def fit(self, X, y=None):
        X = self._check_X(X)
        if self.increasing == 'auto':
            self.increasing = True
            super(_FixedIsotonicRegression, self).fit(X, y)
            score_inc = self.score(X, y)

            self.increasing = False
            super(_FixedIsotonicRegression, self).fit(X, y)
            score_dec = self.score(X, y)
            
            if score_inc > score_dec:
                # Refit because increasing was better
                self.increasing = True
                super(_FixedIsotonicRegression, self).fit(X, y)
            
            # Reset increasing parameter
            self.increasing = 'auto'
        return self
    
    def predict(self, X):
        X = self._check_X(X)
        return super(_FixedIsotonicRegression, self).predict(X)
        
    def _check_X(self, X):
        X = column_or_1d(X)  # Convert single column or 1d vector to 1d vector
        return X


class LeastMonotonicUtility(_ConcreteFunctionalPropertyUtility):
    @classmethod
    def _default_regressor(cls, **kwargs):
        both_kwargs = dict(increasing='auto', out_of_bounds='clip')
        both_kwargs.update(kwargs)
        return _FixedIsotonicRegression(**both_kwargs)


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


class LeastLipschitzUtility(_ConcreteFunctionalPropertyUtility):
    def __init__(self, model, lipschitz_constant=1, loss_metric=None):
        self.model = model
        self.loss_metric = loss_metric
        self.regression_estimator = self._default_regressor(
            lipschitz_constant=lipschitz_constant)

    @classmethod
    def _default_regressor(cls, **kwargs):
        return _LipschitzRegressor(**kwargs)

    @classmethod
    def from_plot_vals(cls, x, y, lipschitz_constant=1, loss_metric=None):
        if loss_metric is None:
            loss_metric = mean_squared_error
        assert np.allclose(np.diff(x), x[1]-x[0]), 'x should be evenly spaced points'

        reg = cls._default_regressor(lipschitz_constant=lipschitz_constant)

        return FunctionalPropertyUtility.from_plot_vals(
            x, y, regression_estimator=reg, loss_metric=loss_metric)


##############################################################################
# Unused utilities (kept for reference but not part of public API)
# May have errors or be broken as well since some things have changed
##############################################################################


class _TotalVariationUtility(_Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s, x0):
        if len(s) < 2:
            return 0
        bandwidth = s[1] - s[0]
        f_out = self.model(X_curve)
        grad_f_out = np.gradient(f_out, s)
        return np.mean(np.abs(grad_f_out))


class _MostJerkUtility(_Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s, x0):
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


class _LeastJerkUtility(_MostJerkUtility):
    def _multiplier(self):
        return -1.0


class _MostCurvatureUtility(_Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s, x0):
        if len(s) < 2:
            return 0
        bandwidth = s[1] - s[0]
        f_out = self.model(X_curve)
        grad_f_out = np.gradient(f_out, s)
        grad2_f_out = np.gradient(grad_f_out, s)
        return self._multiplier() * np.mean(np.abs(grad2_f_out))

    def _multiplier(self):
        return 1.0


class _LeastCurvatureUtility(_MostCurvatureUtility):
    def _multiplier(self):
        return -1.0
    

class _ConditionalLinearUtility(_Utility):
    def __init__(self, model):
        self.model = model

    def _single_curve(self, X_curve, s, x0):
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


class _UniquenessUtility(_Utility):
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


class _GradientUniquenessUtility(_UniquenessUtility):
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
