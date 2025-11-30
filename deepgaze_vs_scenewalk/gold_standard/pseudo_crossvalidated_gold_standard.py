from collections import OrderedDict

import numpy as np
import pysaliency
from scipy.interpolate import RBFInterpolator, griddata


def interpolate_rbf(points, values, XS, YS, kernel="linear", **kwargs):
    def get_values(data):
        ys, xs = np.nonzero(~np.isnan(data))
        zs = data[ys, xs]

        points = np.vstack([xs, ys]).T
        return points, zs

    clean_points = OrderedDict()
    # sometimes we have the same point multiple times, we need to average them
    for point, z in zip(points, values):
        x, y = point
        clean_points.setdefault((x, y), []).append(z)

    for key in clean_points:
        clean_points[key] = np.mean(clean_points[key])

    # xi = np.array(list(clean_points.keys()))
    points = np.array(list(clean_points.keys()))
    values = np.array(list(clean_points.values()))

    xi = np.array((XS, YS)).reshape(2, -1).T
    interpolated = RBFInterpolator(
        points,
        values,
        kernel=kernel,
        **kwargs,
    )(xi).reshape(XS.shape)

    interpolated = np.clip(interpolated, 0, 1)

    return interpolated


def interpolate_rbf_linear_squared(points, values, XS, YS, exponent=2):
    return interpolate_rbf(points, values, XS, YS, kernel="linear") ** exponent


def interpolate_rbf_linear_quadric(points, values, XS, YS):
    return interpolate_rbf_linear_squared(points, values, XS, YS, exponent=4)


def interpolate_linear(points, values, XS, YS):
    def get_values(data):
        ys, xs = np.nonzero(~np.isnan(data))
        zs = data[ys, xs]

        points = np.vstack([xs, ys]).T
        return points, zs

    interpolated = griddata(points, values, (XS, YS), method="linear")
    points, values = get_values(interpolated)
    extrapolated = griddata(points, values, (XS, YS), method="nearest")

    return extrapolated


def interpolate_nearest(points, values, XS, YS):
    extrapolated = griddata(points, values, (XS, YS), method="nearest")

    return extrapolated


def _subject_goldstandard_interpolation(
    stimulus, fixations, subject_models, interpolate=interpolate_rbf_linear_squared
):
    results = []
    for subject, subject_model in subject_models.items():
        indices = fixations.subject == subject
        if not indices.sum():
            continue

        n_subject_fixations = indices.sum()
        n_other_fixations = len(fixations) - n_subject_fixations

        subject_log_density = subject_model.log_density(stimulus)
        xs_source = np.hstack((fixations.x[indices], fixations.x[~indices]))
        ys_source = np.hstack((fixations.y[indices], fixations.y[~indices]))
        zs_source = np.hstack((np.ones(n_subject_fixations), np.zeros(n_other_fixations)))

        height, width = stimulus.shape[:2]
        X = np.arange(width)
        Y = np.arange(height)
        X, Y = np.meshgrid(X, Y)

        points = np.column_stack((xs_source.ravel(), ys_source.ravel()))
        # print(points.shape)
        # print(zs_source.shape)

        # linear_interpolator = LinearNDInterpolator(points, zs_source)
        # interpolation = linear_interpolator(X, Y)
        interpolation = interpolate(points, zs_source, X, Y)

        results.append((subject, subject_log_density, interpolation))
    return results


def fake_crossvalidated_gold_standard_density(
    stimulus, fixations, subject_models, interpolate=interpolate_rbf_linear_squared
):
    results = _subject_goldstandard_interpolation(stimulus, fixations, subject_models, interpolate=interpolate)

    log_densities = np.array([r[1] for r in results])

    interpolations = [r[2] for r in results]
    interpolations = np.array(interpolations)
    sums = np.sum(interpolations, axis=0)

    # handling edge cases
    interpolations[:, sums == 0] = 1
    sums[sums == 0] = len(interpolations)

    interpolations /= np.sum(interpolations, axis=0)

    fake_crossval_log_density = np.log((interpolations * np.exp(log_densities)).sum(axis=0))

    fake_crossval_log_density -= np.log(np.sum(np.exp(fake_crossval_log_density)))

    return fake_crossval_log_density


class PseudoCrossvalidatedGoldStandard(pysaliency.Model):
    """a model class which computes the pseudo averages from the subject specific gold standard models"""

    def __init__(self, stimuli, fixations, subject_models, interpolate=interpolate_rbf_linear_squared, **kwargs):
        super().__init__(**kwargs)
        self.stimuli = stimuli
        self.fixations = fixations
        self.subject_models = subject_models
        self.interpolate = interpolate

    def _log_density(self, stimulus):
        stimulus = pysaliency.datasets.as_stimulus(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus.stimulus_id)

        this_fixations = self.fixations[self.fixations.n == stimulus_index]

        fake_crossval_log_density = fake_crossvalidated_gold_standard_density(
            stimulus, this_fixations, self.subject_models, interpolate=self.interpolate
        )

        return fake_crossval_log_density
