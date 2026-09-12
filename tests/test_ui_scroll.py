from slurmmon.ui import _list_title, _scroll_window


def test_scroll_window_fits_everything_when_short():
    items = list(range(5))
    shown, offset, total = _scroll_window(items, scroll=0, height=40)
    assert shown == items
    assert offset == 0
    assert total == 5


def test_scroll_window_clamps_to_max_offset():
    items = list(range(100))
    # height=10 -> chrome=8 -> 2 visible rows, floored to the 3-row minimum;
    # asking to scroll way past the end should clamp to the last full page.
    shown, offset, total = _scroll_window(items, scroll=9999, height=10)
    assert total == 100
    assert len(shown) == 3
    assert offset == 97
    assert shown == [97, 98, 99]


def test_scroll_window_negative_scroll_clamps_to_zero():
    items = list(range(20))
    shown, offset, total = _scroll_window(items, scroll=-5, height=10)
    assert offset == 0
    assert shown[0] == 0


def test_scroll_window_advances_with_scroll():
    items = list(range(20))
    shown, offset, total = _scroll_window(items, scroll=3, height=10)
    assert offset == 3
    assert shown[0] == 3


def test_list_title_shows_range_when_truncated():
    title = _list_title("Jobs", offset=10, shown=4, total=100)
    assert title == "Jobs (11-14 of 100 -- ↑/↓ PgUp/PgDn to scroll)"


def test_list_title_shows_plain_count_when_everything_fits():
    assert _list_title("Jobs", offset=0, shown=5, total=5) == "Jobs (5)"


def test_list_title_empty():
    assert _list_title("Jobs", offset=0, shown=0, total=0) == "Jobs (none)"


def test_row_delta_shrinks_visible_rows():
    items = list(range(100))
    # height=24 -> chrome=8 -> auto 16 rows; shrinking by 6 should show 10.
    shown, offset, total = _scroll_window(items, scroll=0, height=24, row_delta=-6)
    assert len(shown) == 10


def test_row_delta_cannot_grow_past_terminal_height():
    items = list(range(100))
    # a positive row_delta must not push past what the real terminal fits.
    shown, offset, total = _scroll_window(items, scroll=0, height=24, row_delta=50)
    assert len(shown) == 16  # same as row_delta=0 for this height


def test_row_delta_still_floors_at_minimum_visible_rows():
    items = list(range(100))
    shown, offset, total = _scroll_window(items, scroll=0, height=24, row_delta=-9999)
    assert len(shown) == 3
