import numpy as np
import pandas as pd
import pytest

from src.recency import LOG_PARTS, earlier_click_lists, load_log, repeated_candidates, within_slope


def test_earlier_clicks_are_strictly_earlier_same_user_and_ordered():
    click_users = np.array(["U1", "U1", "U1", "U2"])
    click_times = np.array([30, 10, 20, 5])
    click_items = np.array([3, 1, 2, 9])
    lists = earlier_click_lists(["U1", "U1", "U1", "U2", "U3"], [10, 20, 31, 5, 99], click_users, click_times, click_items)
    assert [x.tolist() for x in lists] == [[], [1], [1, 2, 3], [], []]      # same timestamp is not earlier


def test_repeated_candidates_are_per_impression():
    lists = [np.array([5, 6]), np.array([], dtype=np.int64)]
    out = repeated_candidates(np.array([0, 0, 1]), np.array([5, 7, 5]), lists)
    assert out.tolist() == [True, False, False]                             # item 5 repeats only for impression 0
    assert not repeated_candidates(np.array([0]), np.array([1]), [np.empty(0, dtype=np.int64)]).any()


def test_within_slope_ignores_impression_level_differences():
    groups = np.array([0, 0, 1, 1])
    x = np.array([0.0, 1.0, 10.0, 11.0])
    assert within_slope(groups, x, 2 * x + np.array([5, 5, -3, -3])) == pytest.approx(2.0)
    assert within_slope(np.array([0, 1]), np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0


def test_load_log_reads_only_fit_and_valid(fake_processed):
    assert LOG_PARTS == ("fit", "valid") and "dev" not in LOG_PARTS
    clicks, impressions = load_log(fake_processed)
    full = pd.read_parquet(fake_processed / "impressions.parquet")
    dev_keys = set(full.loc[full["part"].astype(str) == "dev", "imp_key"].astype(str))
    assert dev_keys and not dev_keys & set(impressions.index) and impressions.index.is_unique
    assert len(clicks) == int(full.loc[full["part"].astype(str) != "dev", "label"].sum())


def test_time_matched_partner_is_nearest_other_user():
    from src.recency import time_matched_partner
    times = np.array([10, 11, 12, 50])
    users = np.array(["A", "A", "B", "C"])
    assert time_matched_partner(times, users).tolist() == [2, 2, 1, 2]
    assert time_matched_partner(np.array([1, 2]), np.array(["A", "A"])).tolist() == [0, 1]   # no other user: itself
