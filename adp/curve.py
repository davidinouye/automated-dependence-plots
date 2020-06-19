import warnings
import numpy as np
import scipy.stats

from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array

import torch
from skimage.filters import gaussian
from skimage.transform import rotate
import PIL
from PIL import Image

from .funcs import is_categorical, get_dtype_categories, img2vec, vec2img


def create_coordinate_curves(x0, X, dtypes=None, **kwargs):
    # Create curves for each coordinate
    if dtypes is None:
        dtypes = [np.float for _ in x0]
    sel_categorical = is_categorical(dtypes)
    sel_numeric = ~sel_categorical
    
    # Get curves for each coordinate
    curves = []
    for ii, (is_cat, dtype) in enumerate(zip(sel_categorical, dtypes)):
        v = np.zeros(np.sum(sel_numeric))
        if is_cat:
            z0 = x0[sel_categorical]
            ii_z = np.flatnonzero(np.flatnonzero(sel_categorical) == ii)[0] # Crazy but works
            def create_z(val):
                new_z = z0.copy()
                new_z[ii_z] = val
                return new_z
            Z_list = [create_z(val) for val in range(len(get_dtype_categories(dtype)))]
            if np.all(Z_list[0] == Z_list[1]):
                raise RuntimeError()
            curves.append(CategoricalLinearCurve(x0=x0, v=v, dtypes=dtypes, Z_list=Z_list, **kwargs).fit(X))
        else:
            ii_v = np.flatnonzero(np.flatnonzero(sel_numeric) == ii)[0] # Crazy but works
            v[ii_v] = 1
            curves.append(CategoricalLinearCurve(x0=x0, v=v, dtypes=dtypes, **kwargs).fit(X))
    return np.array(curves)

def create_image_coordinate_curves(x0, X, dtypes=None, curve_types=[]):
    curves = []
    for ii in range(len(curve_types)):
        v = np.zeros(len(curve_types))
        v[ii] = 1.0
        curves.append(ImageTransformationCurve(x0=x0, v=v, dtypes=dtypes, curve_types=curve_types).fit(X))
    return np.array(curves)


def _psi(t, min_scale, max_scale):
    return min_scale + (max_scale - min_scale) * t


def _get_samples(psi_t, v, x0):
    if isinstance(psi_t, torch.Tensor):
        outer = torch.ger
    else: # array-like, default to numpy
        outer = np.outer
    return outer(psi_t, v) + x0


class LinearCurve():
    def __init__(self, x0, v, density_estimator='gaussian', n_grid_density=50, eps='auto'):
        self.x0 = x0
        self.v = v
        self.eps = eps
        self.density_estimator = density_estimator
        self.n_grid_density = n_grid_density
        
    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            dot, outer, sqrt = torch.dot, torch.ger, torch.sqrt
            cast, squeeze, reshape, ndim = lambda x: torch.as_tensor(x, dtype=torch.float64), torch.squeeze, torch.reshape, lambda x: len(x.size())
            mv = torch.mv  # Matrix vector multiplication
        else: # array-like, default to numpy
            dot, outer, sqrt = np.dot, np.outer, np.sqrt
            cast, squeeze, reshape, ndim = np.asarray, np.squeeze, np.reshape, lambda x: x.ndim
            mv = np.dot  # Matrix vector multiplication

        # Setup variables
        t, v, x0, min_vec, max_vec = (cast(a) for a in [t, self.v, self._get_x0(), self.min_, self.max_])
        scalar_input = ndim(t) == 0
        if scalar_input:
            t = reshape(t, (1,))
        assert np.abs(dot(v, v) - 1) < 1e-6, 'v should be approximately a unit vector'

        # Get bounds based on rectangular bounding box
        min_scale, max_scale = self._get_min_max_scale(v, x0, min_vec, max_vec)

        result = _get_samples(_psi(t, min_scale, max_scale), v, x0)

        # Handle scalars
        if scalar_input:
            return np.squeeze(result)
        return result

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.min_ = np.amin(X, axis=0)
        self.max_ = np.amax(X, axis=0)
        if self.density_estimator == 'gaussian':
            X = check_array(X)
            n_samples, n_features = X.shape
            if n_samples < n_features:
                warnings.warn('X given to fit has n_samples < n_features '
                              'so using diagonal covariance Gaussian.')
                self.density_ = GaussianMixture(n_components=1, covariance_type='diag').fit(X)
            else:
                self.density_ = GaussianMixture(n_components=1, covariance_type='full').fit(X)
        elif self.density_estimator is not None:
            self.density_ = clone(self.density_estimator).fit(X)
        else:
            self.density_ = None
        if self.eps == 'auto':
            if self.density_ is not None:
                self.log_eps_ = np.amin(self.density_.score_samples(X))
            else:
                self.log_eps_ = np.log(1e-4)
        else:
            self.log_eps_ = np.log(self.eps)
        return self

    def _get_min_max_scale(self, v, x0, min_vec, max_vec):
        # Check that x0 is in box constraint
        assert np.all(x0 >= min_vec), 'x0 not greater than or equal to min_vec'
        assert np.all(x0 <= max_vec), 'x0 not less than or equal to max_vec'

        # Use min and max to get start and end
        def _solve(a, b, y):
            assert a != 0
            return (y - b)/a

        solved_scale = np.array([
            [_solve(a, b, low), _solve(a, b, high)]
            for a, b, low, high in zip(v, x0, min_vec, max_vec) if a != 0
        ])
        min_scale = np.max([np.min(scale_vec) for scale_vec in solved_scale])
        max_scale = np.min([np.max(scale_vec) for scale_vec in solved_scale])

        # Shrink bounds based on density
        if self.density_ is not None:
            # Get query points for density
            tq = np.linspace(0, 1, num=self.n_grid_density)
            psi_tq = _psi(tq, min_scale, max_scale) 
            Xq = _get_samples(psi_tq, v, x0)

            # Compute log likelihood values for query points
            logL = self.density_.score_samples(Xq)
            high_density_sel = (logL > self.log_eps_)
            if np.sum(high_density_sel) == 0:
                min_scale = 0
                max_scale = 0
            else:
                # Change min and max scale based on which are high density
                min_scale = np.amin(psi_tq[high_density_sel])
                max_scale = np.amax(psi_tq[high_density_sel])

        assert max_scale > min_scale or (max_scale==0 and min_scale==0), 'max_scale not greater than min_scale or min_scale=max_scale=0'
        return min_scale, max_scale

    def get_t0(self):
        if hasattr(self, 'is_only_categorical') and self.is_only_categorical():
            return 0.5
        #TODO try to cleanup somehow (mainly copying from above)
        dot, outer, sqrt = np.dot, np.outer, np.sqrt
        cast, squeeze, reshape, ndim = np.asarray, np.squeeze, np.reshape, lambda x: x.ndim
        mv = np.dot  # Matrix vector multiplication

        # Setup variables
        v, x0, min_vec, max_vec = (cast(a) for a in [self.v, self._get_x0(), self.min_, self.max_])
        assert np.abs(dot(v, v) - 1) < 1e-6, 'v should be approximately a unit vector'

        min_scale, max_scale = self._get_min_max_scale(v, x0, min_vec, max_vec)

        # Solve for t when psi=0
        if max_scale - min_scale == 0:
            t0 = 0.5  # Any t0 will do since no valid values except itself
        else:
            t0 = -min_scale/(max_scale - min_scale)
        assert np.linalg.norm(self(t0)[~self.is_categorical()]-self._get_x0())/np.linalg.norm(self._get_x0()) < 1e-10, 't0 is not correct'
        return t0       

    def _get_x0(self):
        # Need to override in subclass
        return self.x0

    def _check_X(self, X):
        # Need to override in subclass
        return X

    def phi(self, t):
        warnings.warn('Deprecated use of curve.phi() please call curve() directly')
        if isinstance(t, torch.Tensor):
            return torch.as_tensor(self(np.asarray(t)))
        return self(t)

    def is_categorical(self):
        return np.zeros(len(self.x0), dtype=np.bool)

    def is_numeric(self):
        return ~self.is_categorical()

    def _is_changing_categorical(self):
        if hasattr(self, 'Z_list') and self.Z_list is not None:
            Z_mat = np.array(self.Z_list)
            return np.any(Z_mat != self.x0[self.is_categorical()], axis=0)
        else:
            n_categorical = np.sum(self.is_categorical())
            return np.zeros(n_categorical, dtype=np.bool)


class CategoricalLinearCurve(LinearCurve):
    def __init__(self, x0, v, density_estimator='gaussian', n_grid_density=50, eps='auto', dtypes=None, Z_list=None):
        self.x0 = x0
        self.v = v
        self.density_estimator = density_estimator
        self.n_grid_density = n_grid_density
        self.eps = eps

        self.dtypes = dtypes
        self.Z_list = Z_list

    def __call__(self, t, z_idx=-1):
        # Checks of parameters
        x0, v, dtypes, Z_list = self.x0, self.v, self.dtypes, self.Z_list
        assert dtypes is None or len(v) == len(x0) - np.sum(self.is_categorical()), 'len(v) should be the number of numeric columns'
        if Z_list is not None:
            assert np.all(np.asarray(Z_list).shape == np.unique(Z_list, axis=0).shape), 'Z_list should have unique entries'
            assert np.all([len(z) == np.sum(self.is_categorical()) for z in Z_list]), 'len(Z_list[i]) should be equal the number of categories'
        if dtypes is not None and Z_list is None:
            Z_list = [x0[self.is_categorical()]]

        # Handle torch and numpy and check inputs
        if isinstance(t, torch.Tensor):
            cast, tile, ndim, reshape = torch.as_tensor, lambda x, *args: x.repeat(*args), lambda x: len(x.size()), torch.reshape
        else:
            cast, tile, ndim, reshape = np.asarray, np.tile, lambda x: x.ndim, np.reshape
        t = cast(t)
        scalar_input = ndim(t) == 0
        if scalar_input:
            t = reshape(t, (1,))
        if not np.isscalar(z_idx):
            raise ValueError('`z_idx` should be scalar.')

        # Get numeric features
        X = tile(cast(self.x0), (len(t), 1))
        if np.sum(self.is_numeric()) > 0:
            if self.is_only_categorical():
                X[:, self.is_numeric()] = x0[self.is_numeric()]
            else:
                X[:, self.is_numeric()] = super().__call__(t)

        # Get categorical features
        if np.sum(self.is_categorical()) > 0:
            X[:, self.is_categorical()] = Z_list[z_idx]

        # Handle scalar input
        if scalar_input:
            return np.squeeze(X)
        return X

    def is_only_categorical(self):
        return np.all(np.equal(self.v, 0))

    def get_possible_z_idx(self):
        if self.Z_list is None:
            return np.array([-1])
        else:
            return np.arange(len(self.Z_list))

    def _get_x0(self):
        return self.x0[self.is_numeric()]

    def _check_X(self, X):
        return X[:, self.is_numeric()]

    def is_categorical(self):
        if self.dtypes is not None:
            return is_categorical(self.dtypes)
        return np.zeros(len(self.x0), dtype=np.bool)

class ImageTransformationCurve(LinearCurve):
    def __init__(self, x0, v, eps=0.001, dtypes=None, curve_types=[]):
        self.x0 = x0 # flattened image vector
        self.v = v
        self.eps = eps
        self.dtype = dtypes

        self.is_image = True
        self.curve_types = curve_types
        if self.curve_types == []:
            raise ValueError('curve types argument is empty')

        self.name2op = dict()
        self.name2op['brightness'] = self.adjust_brightness
        self.name2op['contrast'] = self.adjust_contrast
        self.name2op['blur'] = self.blur
        self.name2op['rotate'] = self.rotation
        self.name2op['saturate'] = self.saturation
        # NOTE the order of applying the operator depends on the order of the argument
        self.ops = [self.name2op[x] for x in curve_types]

    def __call__(self, t):
        # Handle torch and numpy and check inputs
        if isinstance(t, torch.Tensor):
            cast, tile, ndim, reshape = torch.as_tensor, lambda x, *args: x.repeat(*args), lambda x: len(x.size()), torch.reshape
        else:
            cast, tile, ndim, reshape = np.asarray, np.tile, lambda x: x.ndim, np.reshape
        t = cast(t)
        scalar_input = ndim(t) == 0
        if scalar_input:
            t = reshape(t, (1,))

        params = np.outer(t, self.v)
        X = tile(cast(self.x0), (len(t), 1))
        for i in range(params.shape[0]):
            X[i, :] = img2vec(self.transform_op(vec2img(self.x0), params[i, :]))

        # Handle scalar input
        if scalar_input:
            return np.squeeze(vec2img(X))

        return X

    def fit(self, X):
        self.min_ = np.zeros(self.v.shape)
        self.max_ = np.ones(self.v.shape)
        return self

    def get_t0(self):
        return 0.0

    def is_only_categorical(self):
        return False

    def is_categorical(self):
        return np.zeros(len(self.v), dtype=np.bool)

    def transform_op(self, img, params):
        x = img
        for op, p in zip(self.ops, params):
            x = op(x, p)
        return x

    # define individual transformation operations here
    def blur(self, img, param):
        return gaussian(img.transpose(1,2,0), sigma=3.0 * param, multichannel=True).transpose(2,0,1)

    def adjust_brightness(self, img, param):
        return np.clip(img + param, 0, 1)

    def adjust_contrast(self, img, param):
        return np.clip(img * 6 * (param+1/6), 0, 1)

    def rotation(self, img, param):
        return rotate(img.transpose(1,2,0), param * 180).transpose(2, 0, 1)

    def saturation(self, img, param):
        im = Image.fromarray(np.uint8(img.transpose(1,2,0)*255))
        converter = PIL.ImageEnhance.Color(im)
        img2 = np.array(converter.enhance(np.abs(param-1))) / 255.0
        return img2.transpose(2,0,1)
