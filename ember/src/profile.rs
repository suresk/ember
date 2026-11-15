//! Chapter 24: where a training step spends its time. A stopwatch, one per
//! thread, that the tape reads at every node it records and every node it
//! walks back through, charging the time since the last reading to that node's
//! operation.

use std::cell::RefCell;
use std::fmt;
use std::time::{Duration, Instant};

// @listing profile
/// The time charged to each operation, one row per `(phase, op)`, in the order
/// the operations were first seen.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub rows: Vec<Row>,
}

/// One operation in one phase: how many marks charged it, and how much time.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub phase: &'static str,
    pub op: &'static str,
    pub calls: usize,
    pub time: Duration,
}

/// The running stopwatch: the profile so far, and when it was last read.
struct Stopwatch {
    profile: Profile,
    last: Instant,
}

thread_local! {
    static STOPWATCH: RefCell<Option<Stopwatch>> = const { RefCell::new(None) };
}

/// Run `f` with the stopwatch on, and return its result and the profile. A
/// profile inside `f` takes over the marks until it returns.
pub fn profile<R>(f: impl FnOnce() -> R) -> (R, Profile) {
    let (profile, last) = (Profile::default(), Instant::now());
    let outer = STOPWATCH.replace(Some(Stopwatch { profile, last }));
    let result = f();
    let watch = STOPWATCH.replace(outer).expect("profile: no stopwatch");
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

// @listing rows
impl Profile {
    /// Add one call of `op` in `phase`, taking `time`.
    pub fn add(&mut self, phase: &'static str, op: &'static str, time: Duration) {
        match self
            .rows
            .iter_mut()
            .find(|r| (r.phase, r.op) == (phase, op))
        {
            Some(row) => (row.calls, row.time) = (row.calls + 1, row.time + time),
            None => self.rows.push(Row {
                phase,
                op,
                calls: 1,
                time,
            }),
        }
    }

    /// The calls of `op` in `phase` (0 if it wasn't seen).
    pub fn calls(&self, phase: &str, op: &str) -> usize {
        let row = self.rows.iter().find(|r| (r.phase, r.op) == (phase, op));
        row.map_or(0, |r| r.calls)
    }

    /// The time charged to `phase`, or to everything for `None`.
    pub fn total(&self, phase: Option<&str>) -> Duration {
        let rows = self.rows.iter();
        let rows = rows.filter(|r| phase.is_none_or(|p| p == r.phase));
        rows.map(|r| r.time).sum()
    }
}
// @end

impl Profile {
    /// The time of `op` in `phase`, if it was seen.
    pub fn time(&self, phase: &str, op: &str) -> Option<Duration> {
        let row = self.rows.iter().find(|r| (r.phase, r.op) == (phase, op));
        row.map(|r| r.time)
    }

    /// Every time and count divided by `n`: per step, for a profile of `n` steps.
    /// `n` is a `u32` because that is what a `Duration` divides by.
    pub fn per(&self, n: u32) -> Profile {
        let per = |r: &Row| Row {
            calls: r.calls / n as usize,
            time: r.time / n,
            ..*r
        };
        Profile {
            rows: self.rows.iter().map(per).collect(),
        }
    }
}

/// A table, the slowest operations first, in microseconds, with each one's
/// share of the total.
impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let us = |t: Duration| t.as_secs_f64() * 1e6;
        let total = us(self.total(None)).max(1e-9);
        let mut rows = self.rows.clone();
        rows.sort_by_key(|r| std::cmp::Reverse(r.time));
        writeln!(f, "phase     op                 calls         µs      %")?;
        for Row {
            phase,
            op,
            calls,
            time,
        } in rows
        {
            let (t, share) = (us(time), 100.0 * us(time) / total);
            writeln!(f, "{phase:<9} {op:<16} {calls:>7} {t:>10.1} {share:>6.1}")?;
        }
        write!(f, "total {total:>45.1}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A profile inside a profile takes the marks while it runs, and the outer
    /// one gets them back afterwards.
    #[test]
    fn profiles_nest() {
        let ((), outer) = profile(|| {
            mark("a", "before");
            let ((), inner) = profile(|| mark("b", "inside"));
            assert_eq!((inner.calls("b", "inside"), inner.rows.len()), (1, 1));
            mark("a", "after");
        });
        assert_eq!(outer.calls("a", "before") + outer.calls("a", "after"), 2);
        assert_eq!(outer.calls("b", "inside"), 0);
    }
}
