//! Chapter 24: where a training step spends its time. A stopwatch, one per
//! thread, that the tape reads at every node it records and every node it
//! walks back through, charging the time since the last reading to that node's
//! operation.

use std::cell::RefCell;
use std::fmt;
use std::time::{Duration, Instant};

// @listing profile
/// The time charged to each operation: `(phase, op, calls, total)`, in the
/// order the operations were first seen.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub rows: Vec<(&'static str, &'static str, usize, Duration)>,
}

/// The running stopwatch: the profile so far, and when it was last read.
struct Stopwatch {
    profile: Profile,
    last: Instant,
}

thread_local! {
    static STOPWATCH: RefCell<Option<Stopwatch>> = const { RefCell::new(None) };
}

/// Run `f` with the stopwatch on, and return its result and the profile.
pub fn profile<R>(f: impl FnOnce() -> R) -> (R, Profile) {
    let (profile, last) = (Profile::default(), Instant::now());
    STOPWATCH.set(Some(Stopwatch { profile, last }));
    let result = f();
    let watch = STOPWATCH.take().expect("profile: the stopwatch is gone");
    (result, watch.profile)
}

/// Charge the time since the last mark to `op` in `phase`. Costs a check of a
/// thread-local when no profile is running.
pub fn mark(phase: &'static str, op: &'static str) {
    STOPWATCH.with_borrow_mut(|watch| {
        if let Some(watch) = watch {
            let now = Instant::now();
            watch.profile.add(phase, op, now - watch.last);
            watch.last = now;
        }
    });
}
// @end

impl Profile {
    /// Add one call of `op` in `phase`, taking `time`.
    pub fn add(&mut self, phase: &'static str, op: &'static str, time: Duration) {
        match self.rows.iter_mut().find(|r| (r.0, r.1) == (phase, op)) {
            Some(row) => (row.2, row.3) = (row.2 + 1, row.3 + time),
            None => self.rows.push((phase, op, 1, time)),
        }
    }

    /// The time charged to `phase`, or to everything for `None`.
    pub fn total(&self, phase: Option<&str>) -> Duration {
        let rows = self.rows.iter();
        let rows = rows.filter(|r| phase.is_none_or(|p| p == r.0));
        rows.map(|r| r.3).sum()
    }

    /// The time of `op` in `phase`, if it was seen.
    pub fn time(&self, phase: &str, op: &str) -> Option<Duration> {
        let row = self.rows.iter().find(|r| (r.0, r.1) == (phase, op));
        row.map(|r| r.3)
    }

    /// The calls of `op` in `phase` (0 if it wasn't seen).
    pub fn calls(&self, phase: &str, op: &str) -> usize {
        let row = self.rows.iter().find(|r| (r.0, r.1) == (phase, op));
        row.map_or(0, |r| r.2)
    }
}

/// A table, the slowest operations first, in microseconds, with each one's
/// share of the total.
impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let us = |t: Duration| t.as_secs_f64() * 1e6;
        let total = us(self.total(None)).max(1e-9);
        let mut rows = self.rows.clone();
        rows.sort_by_key(|r| std::cmp::Reverse(r.3));
        writeln!(f, "phase     op                 calls         µs      %")?;
        for (phase, op, calls, time) in rows {
            let share = 100.0 * us(time) / total;
            let t = us(time);
            writeln!(f, "{phase:<9} {op:<16} {calls:>7} {t:>10.1} {share:>6.1}")?;
        }
        write!(f, "total {total:>45.1}")
    }
}

impl Profile {
    /// Every time and count divided by `n`: per step, for a profile of `n`.
    pub fn per(&self, n: u32) -> Profile {
        let row =
            |&(p, o, c, t): &(_, _, usize, Duration)| (p, o, c / n as usize, t / n);
        Profile {
            rows: self.rows.iter().map(row).collect(),
        }
    }
}
