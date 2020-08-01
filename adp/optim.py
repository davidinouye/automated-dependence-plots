import copy
import warnings
import numpy as np

from .funcs import is_categorical, get_dtype_categories
from .curve import create_coordinate_curves, create_image_coordinate_curves
from .plot import plot_curve_vals


def optimize_curve(x0, utility, X, density_estimator='gaussian', dtypes=None, max_numeric_change=2, max_categorical_change=0,
                   curve_eps='auto', n_angle_grid=100, n_utility_grid=50, max_iter=100, max_inner_iter=100,
                   is_image=False, curve_types=None, verbosity=0):
    if curve_types == None:
        curve_types = []
    if dtypes is None:
        dtypes = [np.float for _ in x0]
    def _find_best_curve_and_utility(curves):
        # Get best curve based on utility
        if is_image:
            curves = [curve for curve in curves if np.all(curve.v >= 0)]
        U_list = [utility(curve, n_grid=n_utility_grid) for curve in curves]
        sorted_idx = np.flip(np.argsort(U_list), axis=0)
        best_curve = curves[sorted_idx[0]]
        best_utility = U_list[sorted_idx[0]]
        return best_curve, best_utility
    assert max_categorical_change <= 1, 'Can only handle at most 1 categorical change currently.'

    # Get best initial curve
    sel_categorical = is_categorical(dtypes)
    if is_image:
        curves = create_image_coordinate_curves(x0, X, dtypes=None, curve_types=curve_types)
    else:
        curves = create_coordinate_curves(x0, X, density_estimator=density_estimator, dtypes=dtypes, eps=curve_eps)
        curves = curves[~sel_categorical]  # Only consider numeric curves
    cur_curve, cur_utility = _find_best_curve_and_utility(curves)

    # Setup all possible next curves
    opt_range = np.linspace(-np.pi/2, np.pi/2, num=n_angle_grid, endpoint=False)
    for it in range(max_iter):
        # Handle numeric curves
        for it2 in range(max_inner_iter):
            if np.all(cur_curve.v == 0):
                numeric_curves = np.array([
                    _create_coordinate_curve(cur_curve, i)
                    for i in range(len(cur_curve.v)) if np.count_nonzero(cur_curve.v) < max_numeric_change
                ])
            else:
                numeric_curves = np.array([
                    [
                        [
                            _create_rotated_curve(cur_curve, angle, i, j)
                            for angle in opt_range
                        ]
                        for j, b in enumerate(cur_curve.v) if i != j and (np.count_nonzero(cur_curve.v) < max_numeric_change or b != 0)
                        # Choose only if either already non-zero coordinate or we can add a coordinate (i.e. nonzero < max_numeric_change)
                    ]
                    for i, a in enumerate(cur_curve.v) if a != 0  # Choose from only non-zero coordinates
                ])
            if numeric_curves.size > 0:
                new_curve, new_utility = _find_best_curve_and_utility(numeric_curves.ravel())
            else:
                new_curve, new_utility = cur_curve, cur_utility

            # Break if converged or update
            if new_utility <= cur_utility:
                if verbosity >= 2:
                    print('Exiting inner iteration %d' % it2)
                break
            cur_curve, cur_utility = new_curve, new_utility

        # Handle categorical change
        if max_categorical_change > 0:
            cat_curves = np.concatenate([
                # Create Z_list with [z0, diff_z]
                [
                    _create_cat_curve(cur_curve, x0, feature_i, new_cat_idx, sel_categorical)
                    for new_cat_idx, new_cat_val in enumerate(get_dtype_categories(dtype)) if new_cat_idx != x0[feature_i]
                ]
                for feature_i, (dtype, is_cat) in enumerate(zip(dtypes, sel_categorical)) if is_cat
            ])
            if cat_curves.size > 0:
                new_curve, new_utility = _find_best_curve_and_utility(cat_curves)

        # Break if converged or update
        if new_utility <= cur_utility:
            if verbosity >= 1:
                print('Exiting outer iteration %d' % it)
            break
        cur_curve, cur_utility = new_curve, new_utility
        
    # Check if max iteration reached
    if max_iter > 0 and it == max_iter - 1:
        warnings.warn('Reached max iteration when optimizing, may not have converged')

    # Add both values to Z_list if not equal to original
    #if (hasattr(cur_curve, 'Z_list') 
    #        and cur_curve.Z_list is not None 
    #        and np.any(cur_curve.Z_list[0] != x0[sel_categorical])):
    #    cur_curve.Z_list = [x0[sel_categorical], cur_curve.Z_list[0]]

    return cur_curve


def _optimize_and_plot(X_targets, utility, X_for_bounds=None, max_numeric_change=1, optimize_kwargs=None, plot_kwargs=None):
    # Currently not ready for production since untested so made private
    #'''Optimizes utility over a single target (x0) or a set of targets and then plots the optimal curve.'''
    # Handle single target
    X_targets = np.array(X_targets)
    if X_targets.ndim == 1:
        X_targets = X_targets.reshape(-1, 1)
    if X_for_bounds is None:
        X_for_bounds = X_targets
    if optimize_kwargs is None:
        optimize_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # Optimize curve for each possible point
    best_curves = np.array([
        optimize_curve(x0, utility, X_for_bounds,
                       max_numeric_change=max_numeric_change, **optimize_kwargs)
        for x0 in X_targets
    ])

    # Rerun utility computation to get utility of best curves
    utility_kwargs = {}
    if 'n_utility_grid' in optimize_kwargs:
        utility_kwargs['n_grid'] = optimize_kwargs['n_utility_grid']
    best_utilities = np.array([
        utility(best_curve, **utility_kwargs) 
        for best_curve in best_curves])

    # Choose best sample and best curve
    best_idx = np.argmax(best_utilities)
    best_curve = best_curves[best_idx]

    # Plot best curve and best utility
    plot_curve_vals(curve=best_curve, utility=utility, **plot_kwargs)

    return dict(best_curve=best_curve, best_curves_per_sample=best_curves)


def _create_rotated_curve(curve, rad, i, j):
    assert i != j, 'i and j should be different feature indices'
    Q = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    new_curve = copy.deepcopy(curve)
    feature_scale = new_curve.max_ - new_curve.min_
    v = new_curve.v

    # Operate angle rotation in normalized unit sphere
    v = v / feature_scale
    v = v / np.linalg.norm(v)
    v[[i, j]] = Q.dot(v[[i, j]])
    v = v * feature_scale
    v = v / np.linalg.norm(v)

    new_curve.v = v 
    return new_curve

def _create_cat_curve(curve, x0, feature_i, new_cat_idx, sel_categorical):
    x_diff = x0.copy()
    x_diff[feature_i] = new_cat_idx
    assert x_diff[feature_i] != x0[feature_i], 'Feature values are the same'

    # Create new curve
    new_curve = copy.deepcopy(curve)
    new_curve.Z_list = [x_diff[sel_categorical]]

    return new_curve

def _create_coordinate_curve(curve, i):
    assert np.all(curve.v == 0), 'curve is not all zeros'
    new_curve = copy.deepcopy(curve)
    new_curve.v[i] = 1
    return new_curve

