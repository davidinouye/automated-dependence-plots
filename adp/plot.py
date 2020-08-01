import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker

from .curve import create_coordinate_curves, create_image_coordinate_curves


def plot_curve_vals(curve, model=None, utility=None, feature_labels=None, n_grid=50, ax=None,
                    model_label='Model', other_model_label='auto', target_label='Target', tick_rotation=0,
                    y_bounds=None, replace_space_with_newline=True, legend_kwargs=None, trans=False):
    if ax is None:
        ax = plt.gca()
    if model is None and utility is None:
        raise ValueError('Either model or utility must be set')
    if model is not None and utility is not None:
        raise ValueError('model and utility cannot both be set at the same time')
    if utility is not None:
        model = utility.model

    is_only_categorical = hasattr(curve, 'is_only_categorical') and curve.is_only_categorical()
    
    if is_only_categorical:
        s = np.array([0.1, 0.9])#np.linspace(0, 1, num=2)
    else:
        s = np.linspace(0, 1, num=n_grid)
    try:
        args_list = [(s, z) for z in curve.get_possible_z_idx()]
    except AttributeError:
        args_list = [(s,)]  # Do not need to handle categorical
    X_vals = [curve(*args) for args in args_list]
    f_vals = [model(X_val) for X_val in X_vals]
    if hasattr(curve, 'Z_list') and curve.Z_list is not None:
        cat_changing = True
    else:
        cat_changing = False
    
    # Setup features
    n_features = X_vals[0].shape[1]
    if feature_labels is None:
        feature_labels = ['X%d' % i for i in range(n_features)]
        
    # Get selections for numeric and categorical features
    sel_categorical = curve.is_categorical()
    sel_numeric = ~sel_categorical

    # Figure out which values are changing
    def _get_legend_entry(z_vec):
        sel_changing = curve._is_changing_categorical()
        # Create labels
        labels = np.array(feature_labels)[sel_categorical]
        def get_cat_val(z, dtype):
            try:
                return dtype.categories[int(z)]
            except AttributeError:
                return dtype['categories'][int(z)]
        vals_cat = [get_cat_val(z, dtype) for z, dtype in zip(z_vec, np.array(curve.dtypes)[sel_categorical])]
        labels_and_vals = list(zip(labels, vals_cat))
        # Filter out ones that don't change
        filt_labels_and_vals = [a for a, changing in zip(labels_and_vals, sel_changing) if changing]
        return ','.join(['{}={}'.format(a, b) for a, b in filt_labels_and_vals])
        
    # Plot
    handles = []
    legend = []
    for ii, (X_val, f_val, args) in enumerate(zip(X_vals, f_vals, args_list)):
        if is_only_categorical:
            cur_s = (s + ii) / len(args_list)
        else:
            cur_s = s
        H = ax.plot(cur_s, f_val)
        legend.append(_get_legend_entry(curve.Z_list[args[1]]) if cat_changing else model_label)
        handles.append(H[0])

    # Show utility related things...
    if utility is not None:
        other_f_vals = None
        try:
            # Model contrast utility
            other_f_vals = [utility.get_comparison_model(curve.x0)(X_val) for X_val in X_vals]
        except AttributeError:
            try:
                # Linear model
                other_f_vals = [utility.fit_linear_model(s, f_val)(s)
                                for f_val in f_vals]
            except:
                try:
                    # Partial model
                    other_f_vals = [utility.fit_model(s.reshape(-1, 1), f_val).predict(s.reshape(-1, 1))
                                    for f_val in f_vals]
                except:
                    pass
                else:
                    if other_model_label == 'auto':
                        other_model_label = 'BestReg.'
            else:
                if other_model_label == 'auto':
                    other_model_label = 'BestLin'
        else:
            if other_model_label == 'auto':
                other_model_label = 'ContrastModel'
        prefix = other_model_label

        if other_f_vals is not None:
            for ii, (X_val, f_val, args, prev_line) in enumerate(zip(X_vals, other_f_vals, args_list, handles)):
                if len(handles) == 2 and ii == 0:
                    continue  # Skip the first plot of other values
                if is_only_categorical:
                    cur_s = (s + ii) / len(args_list)
                else:
                    cur_s = s
                H = ax.plot(cur_s, f_val, '--', color=prev_line.get_color())
                handles.append(H[0])
                if len(args_list) > 1:
                    legend.append(prefix + ':' + _get_legend_entry(curve.Z_list[args[1]]) if cat_changing else model_label)
                else:
                    legend.append(prefix)

        # Generalizability
        try:
            other_curves = utility.get_other_curves(curve)
            mean_f = np.mean([np.mean(f_val) for f_val in f_vals])
            other_f_vals_list = [
                utility.mean_shifted_val_list(other_curve, args_list, mean_f)
                for other_curve in other_curves
            ]
        except AttributeError:
            pass
        else:
            # Loop over curves
            for jj, other_f_vals in enumerate(other_f_vals_list):
                # Loop over various categoricals for each curve
                for ii, (f_val, prev_line, args) in enumerate(zip(other_f_vals, handles[:len(other_f_vals)], args_list)):
                    if is_only_categorical:
                        cur_s = (s + ii) / len(args_list)
                    else:
                        cur_s = s
                    H = ax.plot(cur_s, f_val, ':', color=prev_line.get_color())
                    if jj == 0:
                        # Only include one legend entry for multiple generalizability curves
                        handles.append(H[0])
                        label = other_model_label
                        legend_entry = _get_legend_entry(curve.Z_list[args[1]]) if cat_changing else model_label
                        if legend_entry != model_label:
                            # Only add if not just the model_label (i.e. categories are needed)
                            label += ':' + legend_entry
                        legend.append(label)

    # Setup x axis
    if hasattr(curve, 'is_only_categorical') and curve.is_only_categorical():
        # Remove x axis if only categorical
        ax.set_xticks(np.linspace(0, 1, num=len(args_list), endpoint=False) + 0.5 / len(args_list))
        xticklabels = legend[:len(args_list)]
        ax.set_xticklabels(xticklabels, rotation=tick_rotation, ha='right')

        # See if we can simplify axis
        if np.sum(curve._is_changing_categorical()) == 1:
            labels = [lab[:lab.index('=')] for lab in xticklabels]
            uniq_labels = np.unique(labels)
            if len(uniq_labels) == 1:
                xticklabels = [lab[(lab.index('=') + 1):] for lab in xticklabels]
                if len(xticklabels) > 6:
                    ax.set_xticklabels(xticklabels, rotation=20, ha='right')
                else:
                    if replace_space_with_newline:
                        xticklabels = [lab.replace(' ', '\n') for lab in xticklabels]
                    ax.set_xticklabels(xticklabels, rotation=tick_rotation, ha='center')
                ax.set_xlabel(uniq_labels[0])

        new_legend = [model_label]
        new_handles = [Line2D([0], [0], color='k', lw=2, linestyle='-')] 
        # If other lines were plotted then
        if len(legend) > len(args_list):
            # Replace legend and handles
            other_label = legend[len(args_list)]
            try:
                other_label = other_label[:other_label.index(':')]
            except ValueError:
                other_label = '(%s)' % other_model_label
            new_legend.append(other_label)
            new_handles.append(Line2D([0], [0], color='k', lw=2, linestyle=handles[-1].get_linestyle()))
        legend = new_legend
        handles = new_handles
    else:
        # Setup tick labels
        s_ticks = ax.get_xticks()
        #if hasattr(curve, 'is_image'):
        #    s_ticks = s_ticks[1:]

        if not hasattr(curve, 'is_image'):
            X_ticks = curve(s_ticks)
        else:
            X_ticks = np.outer(s_ticks, curve.v)
        X_ticks_numeric = X_ticks[:, sel_numeric]

        # Extract only the ones that change
        sel_changing = curve.v != 0
        #if not hasattr(curve, 'is_image'):
        X_ticks_changing = X_ticks_numeric[:, sel_changing]
        def get_formatter(X_tick):
            F = matplotlib.ticker.ScalarFormatter()
            F.axis = ax.xaxis
            F.set_locs(X_tick)
            return F
        formatters = [get_formatter(X_tick) for X_tick in X_ticks_changing]
        ax.set_xticklabels(['\n'.join([F(xt) for xt in X_tick]) for X_tick, F in zip(X_ticks_changing, formatters)])
        xlabels = np.array(feature_labels)[sel_numeric][sel_changing]
        #else:
        #    xlabels = np.array(feature_labels)[sel_changing]
        xlabel = ', '.join(xlabels)
        if len(xlabels) > 1:
            xlabel = '(%s)' % xlabel
        ax.set_xlabel(xlabel)

    # Plot x0
    if is_only_categorical:
        z0 = curve.x0[sel_categorical]
        i0 = [ii for ii, (s, z_idx) in enumerate(args_list) if np.all(curve.Z_list[z_idx] == z0)][0]
        t0 = (0.5 + i0) / len(args_list)
    else:
        t0 = curve.get_t0()

    if hasattr(curve, 'is_image'):
        f0 = model(curve.x0.reshape(1,-1))
    else:
        f0 = model(curve.x0.reshape(1, -1))[0]
    H = ax.plot([t0], [f0], 'or')
    handles.insert(0, H[0])
    #legend.append(target_label)
    if trans:
        legend.insert(0, target_label)
    else:
        temp = _get_legend_entry(curve.x0[sel_categorical])
        legend.insert(0, target_label + ((', ' + temp) if temp.strip() != '' else ''))

    # Show legend
    if legend_kwargs is not None:
        ax.legend(handles, legend, **legend_kwargs)
    else:
        ax.legend(handles, legend)

    # Setup title
    if utility is not None:
        U = utility(curve, n_grid=n_grid)
        utility_label = type(utility).__name__
        title = '%s=%.2g' % (utility_label, U)
        ax.set_title(title)

    if y_bounds is not None:
        mid = np.mean(y_bounds)
        width = 1.10 * (y_bounds[1] - y_bounds[0])
        yl = np.array([-width/2, width/2]) + mid
        ax.set_ylim(yl)

    plt.tight_layout()
    return ax


def plot_curve(curve, X=None, n_grid=50, ax=None):
    if ax is None:
        ax = plt.gca()
    t = np.linspace(0, 1, num=n_grid)
    if hasattr(curve, 'Z_list'):
        for z_idx in curve.get_possible_z_idx():
            X_curve = curve(t, z_idx)
            ax.plot(X_curve[:, 0], X_curve[:, 1])
    else:
        X_curve = curve(t)
        ax.plot(X_curve[:, 0], X_curve[:, 1])
        
    if X is not None:
        ax.scatter(X[:, 0], X[:, 1], marker='.')
    ax.plot([curve.x0[0]], [curve.x0[1]], 'xr')
    ax.axis('equal')
    return ax


def coordinate_explainer(x0, utility, X, dtypes=None, max_show=3, n_utility_grid=50, axes=None, eps='auto', **kwargs):
    # Calculate utility for each coordinate curve
    curves = create_coordinate_curves(x0, X, dtypes=dtypes, eps=eps)
    U_list = [utility(curve, n_grid=n_utility_grid) for curve in curves]
    sorted_idx = np.flip(np.argsort(U_list), axis=0)

    # Show plots
    show_idx = sorted_idx[:min(max_show, len(sorted_idx))]
    if axes is None:
        _, axes = plt.subplots(len(show_idx), 1, figsize=(8, 5*len(show_idx)), sharey=True)
    if np.array(axes).ndim == 0:
        axes = [axes]
    assert len(axes) >= len(show_idx), 'Not enough axes given to plot requested number of plots'
    for curve, ax in zip(curves[show_idx], axes):
        plot_curve_vals(curve, utility=utility, ax=ax, **kwargs)
    return curves, sorted_idx


def image_coordinate_explainer(x0, utility, X, dtypes=None, curve_types=[], max_show=100, axes=None, **kwargs):
    x0_flat = x0.reshape(x0.shape[0] * x0.shape[1] * x0.shape[2])
    X_flat = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

    curves = create_image_coordinate_curves(x0_flat, X_flat, dtypes=dtypes, curve_types=curve_types)
    U_list = [utility(curve, n_grid=50) for curve in curves]
    sorted_idx = np.flip(np.argsort(U_list), axis=0)

    # Show plots
    show_idx = sorted_idx[:min(max_show, len(sorted_idx))]
    if axes is None:
        _, axes = plt.subplots(len(show_idx), 1, figsize=(7, 2.2*len(show_idx)), sharey=True)
    if np.array(axes).ndim == 0:
        axes = [axes]
    assert len(axes) >= len(show_idx), 'Not enough axes given to plot requested number of plots'
    for curve, ax in zip(curves[show_idx], axes):
        plot_curve_vals(curve, utility=utility, feature_labels=curve_types, ax=ax, **kwargs)


