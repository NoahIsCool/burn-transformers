//! Dependency-free proleptic-Gregorian civil time for chat-template clocks.
//!
//! Hugging Face exposes `strftime_now(format)` to chat templates; Llama 3.x
//! templates use it to print today's date. Formatting needs the civil date,
//! weekday, and day of year of a Unix timestamp, which this module derives
//! with integer arithmetic only (no timezone database, no allocation).
//!
//! # Algorithm
//!
//! The conversion follows Howard Hinnant's `civil_from_days`. Shifting the
//! epoch to 0000-03-01 puts the leap day at the *end* of each computational
//! year, so with `z` days since that origin:
//!
//! ```text
//! era = z / 146097                      (400-year cycles, 146097 days each)
//! doe = z - 146097 * era                day of era,  [0, 146096]
//! yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365    year of era, [0, 399]
//! doy = doe - (365*yoe + yoe/4 - yoe/100)                  day of March-based year, [0, 365]
//! mp  = (5*doy + 2) / 153               March-based month, [0, 11]
//! day = doy - (153*mp + 2)/5 + 1        [1, 31]
//! ```
//!
//! January and February (`mp >= 10`) belong to the following civil year.
//! Timestamps are restricted to years 1 through 9999, the range of Python's
//! `datetime`, so every intermediate value is non-negative and every division
//! is Euclidean.
//!
//! # Verified behavior
//!
//! `verus/tokenization.rs` checks the functions here in place: for every
//! timestamp in range, no arithmetic overflows and each field lies in its
//! calendar range (month 1-12, day 1-31, day of year 1-366, hour 0-23, and so
//! on). Agreement with the calendar itself is covered by golden values from
//! Python's `datetime` and a property test against a day-counting reference.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Earliest supported Unix second: 0001-01-01T00:00:00Z, Python's `datetime.min`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const MIN_UNIX_SECONDS: i64 = -62_135_596_800;
/// Latest supported Unix second: 9999-12-31T23:59:59Z, Python's `datetime.max`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const MAX_UNIX_SECONDS: i64 = 253_402_300_799;
/// Seconds from 0000-03-01T00:00:00Z to the Unix epoch (`719468` days).
#[cfg_attr(verus_keep_ghost, verus_verify)]
const EPOCH_SHIFT_SECONDS: i64 = 62_162_035_200;
/// Seconds in one civil day; leap seconds are not represented, as in Unix time.
#[cfg_attr(verus_keep_ghost, verus_verify)]
const SECONDS_PER_DAY: i64 = 86_400;
/// Days in one 400-year Gregorian cycle.
#[cfg_attr(verus_keep_ghost, verus_verify)]
const DAYS_PER_ERA: i64 = 146_097;

/// Calendar fields of one UTC instant.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CivilTime {
    /// Gregorian year, `1..=9999`.
    pub(crate) year: i64,
    /// Month of year, `1..=12`.
    pub(crate) month: i64,
    /// Day of month, `1..=31`.
    pub(crate) day: i64,
    /// Hour of day, `0..=23`.
    pub(crate) hour: i64,
    /// Minute of hour, `0..=59`.
    pub(crate) minute: i64,
    /// Second of minute, `0..=59`.
    pub(crate) second: i64,
    /// Day of week with Monday as `0` and Sunday as `6`, as in Python's `weekday()`.
    pub(crate) weekday: i64,
    /// Day of year, `1..=366`.
    pub(crate) day_of_year: i64,
}

/// Returns the UTC civil time of `unix_seconds`, or `None` outside years 1-9999.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(civil =>
    ensures
        (MIN_UNIX_SECONDS <= unix_seconds <= MAX_UNIX_SECONDS) == civil.is_some(),
        civil.is_some() ==> civil_time_in_range(civil.unwrap()),
))]
// Verus has no specification for `RangeInclusive::contains`; keep plain comparisons.
#[allow(clippy::manual_range_contains)]
pub(crate) fn civil_time(unix_seconds: i64) -> Option<CivilTime> {
    if MIN_UNIX_SECONDS <= unix_seconds && unix_seconds <= MAX_UNIX_SECONDS {
        Some(civil_time_in_supported_range(unix_seconds))
    } else {
        None
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Every field of a civil time lies inside its calendar range.
pub(crate) open spec fn civil_time_in_range(civil: CivilTime) -> bool {
    &&& -1 <= civil.year <= 10_000
    &&& 1 <= civil.month <= 12
    &&& 1 <= civil.day <= 31
    &&& 0 <= civil.hour <= 23
    &&& 0 <= civil.minute <= 59
    &&& 0 <= civil.second <= 59
    &&& 0 <= civil.weekday <= 6
    &&& 1 <= civil.day_of_year <= 366
}

} // verus!

/// Converts an in-range Unix second to civil fields with Hinnant's algorithm.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(civil =>
    requires
        MIN_UNIX_SECONDS <= unix_seconds <= MAX_UNIX_SECONDS,
    ensures
        civil_time_in_range(civil),
))]
fn civil_time_in_supported_range(unix_seconds: i64) -> CivilTime {
    // Shifting to 0000-03-01 makes every quantity below non-negative.
    let shifted = unix_seconds + EPOCH_SHIFT_SECONDS;
    let days = shifted / SECONDS_PER_DAY;
    let second_of_day = shifted % SECONDS_PER_DAY;
    let era = days / DAYS_PER_ERA;
    let day_of_era = days - era * DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_march_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let march_month = (5 * day_of_march_year + 2) / 153;
    let day = day_of_march_year - (153 * march_month + 2) / 5 + 1;
    let march_year = year_of_era + era * 400;
    let (year, month, day_of_year) = if march_month < 10 {
        // March through December: 59 days (60 in leap years) precede March 1.
        let leap = is_leap_year(march_year);
        let february_extra = if leap { 1 } else { 0 };
        (
            march_year,
            march_month + 3,
            day_of_march_year + 59 + february_extra + 1,
        )
    } else {
        // January and February: March 1 through December 31 spans 306 days.
        (march_year + 1, march_month - 9, day_of_march_year - 306 + 1)
    };
    CivilTime {
        year,
        month,
        day,
        hour: second_of_day / 3600,
        minute: second_of_day % 3600 / 60,
        second: second_of_day % 60,
        // Day 0 of the shifted origin, 0000-03-01, was a Wednesday (Monday = 0).
        weekday: (days + 2) % 7,
        day_of_year,
    }
}

/// Gregorian leap-year rule: divisible by 4, except centuries not divisible by 400.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(leap =>
    requires
        0 <= year,
))]
pub(crate) fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

#[cfg(test)]
#[path = "calendar_tests.rs"]
mod tests;
