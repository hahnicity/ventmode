import pandas as pd

from ventmode.main import merge_periods_with_low_time_thresh


def test_merge_periods_with_low_time_thresh():
    preds = pd.Series([0] * 1000 + [1] * 5 + [3] * 1000 + [6] * 5)
    patient = pd.Series(['foo'] * 2010)
    # each breath is approx 3 seconds
    abs_bs = pd.Series([pd.to_datetime('2019-01-01 01:01:01') + pd.Timedelta(seconds=3*i) for i in range(2010)])
    new_preds = merge_periods_with_low_time_thresh(preds, patient, abs_bs, pd.Timedelta(minutes=5))
    assert (new_preds == pd.Series([0] * 1005 + [3] * 1005)).all()


def test_merge_periods_with_low_time_thresh2():
    preds = pd.Series([0] * 1000 + [1] * 5 + [3] * 1000 + [6] * 5 + [1] * 5 + [0] * 500)
    patient = pd.Series(['foo'] * 2515)
    # each breath is approx 3 seconds
    abs_bs = pd.Series([pd.to_datetime('2019-01-01 01:01:01') + pd.Timedelta(seconds=3*i) for i in range(2515)])
    new_preds = merge_periods_with_low_time_thresh(preds, patient, abs_bs, pd.Timedelta(minutes=5))
    assert (new_preds == pd.Series([0] * 1005 + [3] * 1010 + [0] * 500)).all()


def test_merge_periods_with_low_time_thresh_low_boundary():
    preds = pd.Series([0] * 1000 + [1] * 10 + [3] * 1000 + [6] * 5 + [1] * 5 + [0] * 500)
    patient = pd.Series(['foo'] * 2515)
    # each breath is approx 3 seconds
    abs_bs = pd.Series([pd.to_datetime('2019-01-01 01:01:01') + pd.Timedelta(seconds=3*i) for i in range(2515)])
    new_preds = merge_periods_with_low_time_thresh(preds, patient, abs_bs, pd.Timedelta(seconds=20))
    assert (new_preds == pd.Series([0] * 1000 + [1] * 10 + [3] * 1010 + [0] * 500)).all()
