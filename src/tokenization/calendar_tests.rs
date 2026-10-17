//! Calendar tests against Python `datetime` goldens and a day-counting reference.

use proptest::prelude::*;

use super::*;

/// Builds an expected civil time from literal fields.
#[allow(clippy::too_many_arguments)]
fn civil(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
    weekday: i64,
    day_of_year: i64,
) -> CivilTime {
    CivilTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
        weekday,
        day_of_year,
    }
}

/// Golden instants from `datetime.fromtimestamp(t, timezone.utc)` in Python 3.11.
#[test]
fn matches_python_datetime_goldens() {
    for (seconds, expected) in [
        (0, civil(1970, 1, 1, 0, 0, 0, 3, 1)),
        (-1, civil(1969, 12, 31, 23, 59, 59, 2, 365)),
        (-31_622_400, civil(1968, 12, 31, 0, 0, 0, 1, 366)),
        (951_782_400, civil(2000, 2, 29, 0, 0, 0, 1, 60)),
        (1_721_999_109, civil(2024, 7, 26, 13, 5, 9, 4, 208)),
        (4_102_444_799, civil(2099, 12, 31, 23, 59, 59, 3, 365)),
        (MAX_UNIX_SECONDS, civil(9999, 12, 31, 23, 59, 59, 4, 365)),
        (MIN_UNIX_SECONDS, civil(1, 1, 1, 0, 0, 0, 0, 1)),
    ] {
        assert_eq!(
            civil_time(seconds),
            Some(expected),
            "unix seconds {seconds}"
        );
    }
}

/// Instants outside Python's `datetime` range are rejected rather than wrapped.
#[test]
fn rejects_instants_outside_years_one_through_9999() {
    assert_eq!(civil_time(MIN_UNIX_SECONDS - 1), None);
    assert_eq!(civil_time(MAX_UNIX_SECONDS + 1), None);
    assert_eq!(civil_time(i64::MIN), None);
    assert_eq!(civil_time(i64::MAX), None);
}

/// The Gregorian leap rule handles ordinary, century, and 400-year boundaries.
#[test]
fn leap_years_follow_the_gregorian_rule() {
    for (year, leap) in [
        (1, false),
        (4, true),
        (1900, false),
        (2000, true),
        (2023, false),
        (2024, true),
        (2100, false),
        (2400, true),
    ] {
        assert_eq!(is_leap_year(year), leap, "{year}");
    }
}

/// Straightforward reference: count whole days forward or backward from 1970-01-01.
fn reference_civil_time(unix_seconds: i64) -> CivilTime {
    let mut days = unix_seconds.div_euclid(86_400);
    let second_of_day = unix_seconds.rem_euclid(86_400);
    let weekday = (days + 3).rem_euclid(7);
    let year_length = |year: i64| if is_leap_year(year) { 366 } else { 365 };
    let mut year = 1970;
    while days < 0 {
        year -= 1;
        days += year_length(year);
    }
    while days >= year_length(year) {
        days -= year_length(year);
        year += 1;
    }
    let day_of_year = days + 1;
    let mut month = 1;
    let february = if is_leap_year(year) { 29 } else { 28 };
    for length in [31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] {
        if days < length {
            break;
        }
        days -= length;
        month += 1;
    }
    CivilTime {
        year,
        month,
        day: days + 1,
        hour: second_of_day / 3600,
        minute: second_of_day % 3600 / 60,
        second: second_of_day % 60,
        weekday,
        day_of_year,
    }
}

/// Pre-epoch clock readings round toward negative infinity.
#[test]
fn pre_epoch_seconds_round_down() {
    assert_eq!(seconds_before_epoch(0, false), 0);
    assert_eq!(seconds_before_epoch(0, true), -1);
    assert_eq!(seconds_before_epoch(5, true), -6);
    assert_eq!(seconds_before_epoch(i64::MAX, true), i64::MIN);
}

proptest! {
    /// Hinnant's closed form agrees with day counting across the modern range.
    #[test]
    fn closed_form_matches_day_counting(seconds in -6_000_000_000_i64..8_000_000_000) {
        prop_assert_eq!(civil_time(seconds), Some(reference_civil_time(seconds)));
    }

    /// Every supported instant yields fields inside their calendar ranges.
    #[test]
    fn fields_stay_in_calendar_ranges(seconds in MIN_UNIX_SECONDS..=MAX_UNIX_SECONDS) {
        let civil = civil_time(seconds).expect("in-range instant");
        prop_assert!((1..=9999).contains(&civil.year));
        prop_assert!((1..=12).contains(&civil.month));
        prop_assert!((1..=31).contains(&civil.day));
        prop_assert!((1..=366).contains(&civil.day_of_year));
        prop_assert!((0..=6).contains(&civil.weekday));
        prop_assert!((0..=23).contains(&civil.hour));
        prop_assert!((0..=59).contains(&civil.minute));
        prop_assert!((0..=59).contains(&civil.second));
    }
}
