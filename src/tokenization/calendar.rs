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
//! `verus/tokenization.rs` checks the functions here in place against a
//! transcription of CPython's own calendar helpers (`_is_leap`,
//! `_days_before_year`, `_days_in_month`, `_days_before_month`, `_ymd2ord`).
//! For every timestamp in years 1 through 9999 the proofs establish, beyond
//! the absence of overflow, that the fields are exactly Python's `datetime`
//! reading of the instant in UTC:
//!
//! - the date is valid (`1 <= day <= days_in_month(year, month)`, year
//!   1-9999) and its proleptic Gregorian ordinal is `floor(t / 86400) +
//!   719163`, the day number of the instant counted from 1970-01-01;
//! - `hour * 3600 + minute * 60 + second` is `t mod 86400`, each field in range;
//! - the weekday is Python's `weekday()`, `(ordinal + 6) mod 7`;
//! - the day of year is `_days_before_month(year, month) + day`;
//! - `is_leap_year` is Python's `_is_leap` for non-negative years.
//!
//! The crux, that Hinnant's year-of-era estimate names the March-based year
//! containing every day of an era, is proved from the estimate's monotonicity
//! and an evaluation (`by (compute_only)`) of its first and last day for all
//! 400 years of an era. Golden values from Python and a day-counting property
//! test remain as executable cross-checks.

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
    /// Day of month, `1..=days_in_month(year, month)`.
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
///
/// Verified: the result exists exactly for the instants of years 1 through
/// 9999, and then every field is the one Python's `datetime` reports for
/// that instant in UTC (see [`civil_time_in_supported_range`]).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(civil =>
    ensures
        (MIN_UNIX_SECONDS <= unix_seconds <= MAX_UNIX_SECONDS) == civil.is_some(),
        civil.is_some() ==> is_civil_time_of(unix_seconds as int, civil.unwrap()),
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

// Specification: Python's `datetime` calendar.
//
// The functions below transcribe CPython's `datetime.py` helpers
// (`_is_leap`, `_days_before_year`, `_days_in_month`, `_days_before_month`,
// `_ymd2ord`), which define the proleptic Gregorian ordinal with 0001-01-01
// as day 1. They are the specification: Hinnant's closed form above is
// proved to land on the date whose ordinal is the instant's day number.

#[cfg(verus_keep_ghost)]
verus! {

/// Python's `_is_leap(year)`: divisible by 4, except centuries not divisible by 400.
pub(crate) open spec fn gregorian_leap(year: int) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Python's `_days_before_year(year)`: days from 0001-01-01 to January 1 of `year`.
pub(crate) open spec fn days_before_year(year: int) -> int {
    let y = year - 1;
    y * 365 + y / 4 - y / 100 + y / 400
}

/// Days before the first of `month` (1-12) in a common year.
pub(crate) open spec fn common_days_before_month(month: int) -> int {
    if month <= 1 {
        0
    } else if month == 2 {
        31
    } else if month == 3 {
        59
    } else if month == 4 {
        90
    } else if month == 5 {
        120
    } else if month == 6 {
        151
    } else if month == 7 {
        181
    } else if month == 8 {
        212
    } else if month == 9 {
        243
    } else if month == 10 {
        273
    } else if month == 11 {
        304
    } else {
        334
    }
}

/// Python's `_days_before_month(year, month)`: days in `year` before the first of `month`.
pub(crate) open spec fn days_before_month(year: int, month: int) -> int {
    if month > 2 && gregorian_leap(year) {
        common_days_before_month(month) + 1
    } else {
        common_days_before_month(month)
    }
}

/// Python's `_days_in_month(year, month)`.
pub(crate) open spec fn days_in_month(year: int, month: int) -> int {
    if month == 2 {
        if gregorian_leap(year) {
            29
        } else {
            28
        }
    } else if month == 4 || month == 6 || month == 9 || month == 11 {
        30
    } else {
        31
    }
}

/// Python's `_ymd2ord(year, month, day)`: the proleptic Gregorian ordinal, 0001-01-01 being 1.
pub(crate) open spec fn ordinal(year: int, month: int, day: int) -> int {
    days_before_year(year) + days_before_month(year, month) + day
}

/// Python's `date(1970, 1, 1).toordinal()`: the ordinal of the Unix epoch.
pub(crate) open spec fn unix_epoch_ordinal() -> int {
    719_163
}

/// `civil` is the UTC calendar reading of `unix_seconds` that Python's `datetime` gives.
///
/// The date is valid (`1 <= day <= days_in_month`) and its ordinal is the
/// instant's floor day number counted from the epoch's ordinal; the time of
/// day decomposes the floor remainder; the weekday is Python's
/// `(ordinal + 6) % 7` (Monday = 0); and the day of year is Python's
/// `_days_before_month + day`.
pub(crate) open spec fn is_civil_time_of(unix_seconds: int, civil: CivilTime) -> bool {
    let date = ordinal(civil.year as int, civil.month as int, civil.day as int);
    &&& 1 <= civil.year <= 9999
    &&& 1 <= civil.month <= 12
    &&& 1 <= civil.day <= days_in_month(civil.year as int, civil.month as int)
    &&& date == unix_seconds / 86_400 + unix_epoch_ordinal()
    &&& 0 <= civil.hour <= 23
    &&& 0 <= civil.minute <= 59
    &&& 0 <= civil.second <= 59
    &&& civil.hour * 3600 + civil.minute * 60 + civil.second == unix_seconds % 86_400
    &&& civil.weekday == (date + 6) % 7
    &&& civil.day_of_year == days_before_month(civil.year as int, civil.month as int) + civil.day
}

// Proof: Hinnant's closed form computes that date.

/// Days from 0000-03-01 to March 1 of `march_year`: Python's `_days_before_year(march_year + 1)`.
pub(crate) open spec fn march_year_start(march_year: int) -> int {
    march_year * 365 + march_year / 4 - march_year / 100 + march_year / 400
}

/// Hinnant's estimate of the March-based year containing day `day_of_era` of a 400-year era.
pub(crate) open spec fn year_of_era_estimate(day_of_era: int) -> int {
    (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365
}

/// First day of March-based month `march_month` (0 = March) within its March-based year.
pub(crate) open spec fn march_month_start(march_month: int) -> int {
    (153 * march_month + 2) / 5
}

/// A calendar year is 365 days, plus one when it is a leap year.
pub(crate) proof fn lemma_year_length(year: int)
    ensures
        days_before_year(year + 1) - days_before_year(year)
            == 365 + (if gregorian_leap(year) { 1int } else { 0 }),
{
}

/// Hinnant's year estimate never decreases as the day of the era grows.
pub(crate) proof fn lemma_year_estimate_monotone(earlier: int, later: int)
    requires
        0 <= earlier <= later <= 146_096,
    ensures
        year_of_era_estimate(earlier) <= year_of_era_estimate(later),
{
}

/// Every year of an era: the estimate is exact on the year's first and last day.
pub(crate) open spec fn estimate_exact_on_years(first: int, last: int) -> bool
    decreases last - first,
{
    if last - first <= 0 {
        true
    } else if last - first == 1 {
        &&& march_year_start(first) < march_year_start(first + 1)
        &&& year_of_era_estimate(march_year_start(first)) == first
        &&& year_of_era_estimate(march_year_start(first + 1) - 1) == first
    } else {
        let middle = first + (last - first) / 2;
        estimate_exact_on_years(first, middle) && estimate_exact_on_years(middle, last)
    }
}

/// Extracts one year's endpoint facts from [`estimate_exact_on_years`].
pub(crate) proof fn lemma_estimate_exact_on_year(first: int, last: int, year: int)
    requires
        estimate_exact_on_years(first, last),
        first <= year < last,
    ensures
        march_year_start(year) < march_year_start(year + 1),
        year_of_era_estimate(march_year_start(year)) == year,
        year_of_era_estimate(march_year_start(year + 1) - 1) == year,
    decreases last - first,
{
    if last - first > 1 {
        let middle = first + (last - first) / 2;
        if year < middle {
            lemma_estimate_exact_on_year(first, middle, year);
        } else {
            lemma_estimate_exact_on_year(middle, last, year);
        }
    }
}

/// Hinnant's estimate names the March-based year that contains `day_of_era`.
///
/// Searches upward from `year`, whose start is at most `day_of_era`; on the
/// containing year, the endpoint facts (checked by evaluation for all 400
/// years) and monotonicity pin the estimate.
pub(crate) proof fn lemma_year_of_era_from(day_of_era: int, year: int)
    requires
        0 <= year < 400,
        march_year_start(year) <= day_of_era < 146_097,
        estimate_exact_on_years(0, 400),
    ensures
        0 <= year_of_era_estimate(day_of_era) < 400,
        march_year_start(year_of_era_estimate(day_of_era)) <= day_of_era,
        day_of_era < march_year_start(year_of_era_estimate(day_of_era) + 1),
    decreases 400 - year,
{
    lemma_estimate_exact_on_year(0, 400, year);
    if day_of_era < march_year_start(year + 1) {
        lemma_year_estimate_monotone(march_year_start(year), day_of_era);
        lemma_year_estimate_monotone(day_of_era, march_year_start(year + 1) - 1);
    } else {
        assert(march_year_start(400) == 146_097);
        lemma_year_of_era_from(day_of_era, year + 1);
    }
}

/// Hinnant's estimate names the March-based year that contains every day of an era.
pub(crate) proof fn lemma_year_of_era(day_of_era: int)
    requires
        0 <= day_of_era < 146_097,
    ensures
        0 <= year_of_era_estimate(day_of_era) < 400,
        march_year_start(year_of_era_estimate(day_of_era)) <= day_of_era,
        day_of_era < march_year_start(year_of_era_estimate(day_of_era) + 1),
{
    assert(estimate_exact_on_years(0, 50)) by (compute_only);
    assert(estimate_exact_on_years(50, 100)) by (compute_only);
    assert(estimate_exact_on_years(100, 150)) by (compute_only);
    assert(estimate_exact_on_years(150, 200)) by (compute_only);
    assert(estimate_exact_on_years(200, 250)) by (compute_only);
    assert(estimate_exact_on_years(250, 300)) by (compute_only);
    assert(estimate_exact_on_years(300, 350)) by (compute_only);
    assert(estimate_exact_on_years(350, 400)) by (compute_only);
    assert(estimate_exact_on_years(0, 100));
    assert(estimate_exact_on_years(100, 200));
    assert(estimate_exact_on_years(200, 300));
    assert(estimate_exact_on_years(300, 400));
    assert(estimate_exact_on_years(0, 200));
    assert(estimate_exact_on_years(200, 400));
    assert(estimate_exact_on_years(0, 400));
    lemma_year_of_era_from(day_of_era, 0);
}

/// The March-based month formula partitions a March-based year into its twelve months.
pub(crate) proof fn lemma_march_month(day_of_march_year: int)
    requires
        0 <= day_of_march_year <= 365,
    ensures
        0 <= (5 * day_of_march_year + 2) / 153 <= 11,
        march_month_start((5 * day_of_march_year + 2) / 153) <= day_of_march_year,
        (5 * day_of_march_year + 2) / 153 < 11
            ==> day_of_march_year < march_month_start((5 * day_of_march_year + 2) / 153 + 1),
{
}

/// 400 March-based years are exactly 146 097 days, so eras shift year starts uniformly.
pub(crate) proof fn lemma_era_shift(year_of_era: int, era: int)
    requires
        0 <= year_of_era <= 400,
        0 <= era,
    ensures
        march_year_start(year_of_era + era * 400) == march_year_start(year_of_era) + era * 146_097,
        gregorian_leap(year_of_era + era * 400) == gregorian_leap(year_of_era),
{
}

/// The March-based month table: start offsets, civil month lengths, and days before each month.
pub(crate) proof fn lemma_march_month_table(march_month: int, year: int)
    requires
        0 <= march_month <= 11,
    ensures
        march_month < 10 ==> common_days_before_month(march_month + 3) == 59 + march_month_start(
            march_month,
        ),
        march_month < 10 ==> march_month_start(march_month + 1) - march_month_start(march_month)
            == days_in_month(year, march_month + 3),
        march_month == 10 ==> march_month_start(10) == 306 && march_month_start(11) == 337,
        march_month == 11 ==> march_month_start(11) == 337,
{
    if march_month == 0 {
    } else if march_month == 1 {
    } else if march_month == 2 {
    } else if march_month == 3 {
    } else if march_month == 4 {
    } else if march_month == 5 {
    } else if march_month == 6 {
    } else if march_month == 7 {
    } else if march_month == 8 {
    } else if march_month == 9 {
    } else if march_month == 10 {
    } else {
    }
}

/// Hinnant's era and year-of-era steps place day `z` inside its March-based year.
///
/// With `z = era * 146097 + day_of_era`, the March-based year is
/// `march_year = year_of_era + 400 * era` and `z` is day `day_of_march_year`
/// of it, which is less than that year's length (366 exactly when the
/// following February has 29 days).
pub(crate) proof fn lemma_hinnant_year(
    z: int,
    era: int,
    day_of_era: int,
    year_of_era: int,
    day_of_march_year: int,
    march_year: int,
)
    requires
        0 <= z,
        era == z / 146_097,
        day_of_era == z - era * 146_097,
        year_of_era == year_of_era_estimate(day_of_era),
        day_of_march_year == day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100),
        march_year == year_of_era + era * 400,
    ensures
        0 <= march_year,
        z == march_year_start(march_year) + day_of_march_year,
        0 <= day_of_march_year,
        day_of_march_year < 365 + (if gregorian_leap(march_year + 1) { 1int } else { 0 }),
{
    lemma_year_of_era(day_of_era);
    lemma_era_shift(year_of_era, era);
    lemma_era_shift(year_of_era + 1, era);
    lemma_year_length(year_of_era + 1);
}

/// Hinnant's month and day steps, for March through December, give Python's date of day `z`.
pub(crate) proof fn lemma_hinnant_spring(
    z: int,
    day_of_march_year: int,
    march_month: int,
    day: int,
    march_year: int,
)
    requires
        306 <= z <= 3_652_364,
        0 <= march_year,
        z == march_year_start(march_year) + day_of_march_year,
        0 <= day_of_march_year,
        day_of_march_year < 365 + (if gregorian_leap(march_year + 1) { 1int } else { 0 }),
        march_month == (5 * day_of_march_year + 2) / 153,
        day == day_of_march_year - march_month_start(march_month) + 1,
        march_month < 10,
    ensures
        1 <= march_year <= 9999,
        1 <= day <= days_in_month(march_year, march_month + 3),
        ordinal(march_year, march_month + 3, day) == z - 305,
        day_of_march_year + 59 + (if gregorian_leap(march_year) { 1int } else { 0 }) + 1
            == days_before_month(march_year, march_month + 3) + day,
{
    lemma_march_month(day_of_march_year);
    lemma_march_month_table(march_month, march_year);
    lemma_year_length(march_year);
    if march_year >= 10_000 {
        assert(march_year_start(march_year) >= 3_652_400);
    }
}

/// Hinnant's month and day steps, for January and February, give Python's date of day `z`.
pub(crate) proof fn lemma_hinnant_winter(
    z: int,
    day_of_march_year: int,
    march_month: int,
    day: int,
    march_year: int,
)
    requires
        306 <= z <= 3_652_364,
        0 <= march_year,
        z == march_year_start(march_year) + day_of_march_year,
        0 <= day_of_march_year,
        day_of_march_year < 365 + (if gregorian_leap(march_year + 1) { 1int } else { 0 }),
        march_month == (5 * day_of_march_year + 2) / 153,
        day == day_of_march_year - march_month_start(march_month) + 1,
        march_month >= 10,
    ensures
        1 <= march_year + 1 <= 9999,
        1 <= day <= days_in_month(march_year + 1, march_month - 9),
        ordinal(march_year + 1, march_month - 9, day) == z - 305,
        day_of_march_year - 306 + 1 == days_before_month(march_year + 1, march_month - 9) + day,
{
    lemma_march_month(day_of_march_year);
    lemma_march_month_table(march_month, march_year + 1);
    if march_year >= 9999 {
        assert(march_year_start(march_year) >= 3_652_059);
    }
}

/// The time of day and weekday of [`civil_time_in_supported_range`] are Python's.
pub(crate) proof fn lemma_time_of_day(unix_seconds: int, shifted: int, days: int, second_of_day: int)
    requires
        shifted == unix_seconds + 62_162_035_200,
        0 <= shifted,
        days == shifted / 86_400,
        second_of_day == shifted % 86_400,
    ensures
        days == unix_seconds / 86_400 + 719_468,
        second_of_day == unix_seconds % 86_400,
        0 <= second_of_day / 3600 <= 23,
        0 <= second_of_day % 3600 / 60 <= 59,
        0 <= second_of_day % 60 <= 59,
        second_of_day / 3600 * 3600 + second_of_day % 3600 / 60 * 60 + second_of_day % 60
            == second_of_day,
        (days + 2) % 7 == (days - 305 + 6) % 7,
{
}

} // verus!

/// Converts an in-range Unix second to civil fields with Hinnant's algorithm.
///
/// Verified: every field equals Python's `datetime` reading of the instant in
/// UTC, as [`is_civil_time_of`] states: the date is valid and its proleptic
/// Gregorian ordinal is the floor day number plus the epoch's ordinal, the
/// time of day decomposes the floor remainder, the weekday is Python's
/// `weekday()`, and the day of year is Python's `tm_yday`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(civil =>
    requires
        MIN_UNIX_SECONDS <= unix_seconds <= MAX_UNIX_SECONDS,
    ensures
        is_civil_time_of(unix_seconds as int, civil),
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
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_time_of_day(unix_seconds as int, shifted as int, days as int, second_of_day as int);
        lemma_hinnant_year(
            days as int,
            era as int,
            day_of_era as int,
            year_of_era as int,
            day_of_march_year as int,
            march_year as int,
        );
        if march_month < 10 {
            lemma_hinnant_spring(
                days as int,
                day_of_march_year as int,
                march_month as int,
                day as int,
                march_year as int,
            );
        } else {
            lemma_hinnant_winter(
                days as int,
                day_of_march_year as int,
                march_month as int,
                day as int,
                march_year as int,
            );
        }
    }
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
///
/// Verified: equals Python's `_is_leap` ([`gregorian_leap`]) for every
/// non-negative year.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(leap =>
    requires
        0 <= year,
    ensures
        leap == gregorian_leap(year as int),
))]
pub(crate) fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

#[cfg(test)]
#[path = "calendar_tests.rs"]
mod tests;
