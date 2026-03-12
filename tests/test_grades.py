"""Tests for grade sorting utility."""

from climb_sensei.grades import grade_sort_key


def test_hueco_ordering():
    grades = ["V4", "V0", "V10", "V2", "V7"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "hueco"))
    assert sorted_grades == ["V0", "V2", "V4", "V7", "V10"]


def test_french_ordering():
    grades = ["6b+", "5a", "7a", "6a", "8a+"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "french"))
    assert sorted_grades == ["5a", "6a", "6b+", "7a", "8a+"]


def test_yds_ordering():
    grades = ["5.11a", "5.9", "5.10d", "5.12b"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "yds"))
    assert sorted_grades == ["5.9", "5.10d", "5.11a", "5.12b"]


def test_font_ordering():
    grades = ["6A+", "4", "7A", "5+", "8A"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "font"))
    assert sorted_grades == ["4", "5+", "6A+", "7A", "8A"]


def test_unknown_grade_sorts_to_end():
    grades = ["V4", "???", "V2"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "hueco"))
    assert sorted_grades == ["V2", "V4", "???"]


def test_unknown_system_sorts_to_end():
    assert grade_sort_key("V4", "unknown") == (999, 999)
