from dtwco.warping.core import dtw
import numpy as np


def _find_per_breath_dtw_score(prev_pressure_flow_waves, breath):
    # compare the last n_breaths to current breath to compute DTW score
    score = 0
    for pressure, flow in prev_pressure_flow_waves:
        score += dtw(pressure, breath['pressure'])
        score += dtw(flow, breath['flow'])
    return score / len(prev_pressure_flow_waves)


def dtw_file_analyze(generator, n_breaths, rolling_av_len):
    pressure_flow_waves = []
    dtw_scores = [np.nan] * n_breaths
    rel_bns = []
    for breath in generator:
        rel_bns.append(breath['rel_bn'])
        if len(pressure_flow_waves) == (n_breaths+1):
            pressure_flow_waves.pop(0)

        if len(pressure_flow_waves) < (n_breaths):
            pressure_flow_waves.append((breath['pressure'], breath['flow']))
            continue

        dtw_scores.append(_find_per_breath_dtw_score(pressure_flow_waves, breath))
        pressure_flow_waves.append((breath['pressure'], breath['flow']))
    rolling_av = np.convolve(dtw_scores, np.ones((rolling_av_len,))/rolling_av_len, mode='valid')
    return np.append([np.nan]*(rolling_av_len-1), rolling_av), rel_bns
