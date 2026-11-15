//! `cargo xtask check 2` runs Chapter 2's tests (the chapter's examples and each of
//! its challenges) and says which pass. `cargo xtask check 2.3` runs one challenge,
//! and `cargo xtask check all` every chapter. Add `--crate <dir>` to run them against
//! your own crate. See `check.rs`.
//! `cargo xtask bench matmul [sizes]` and `cargo xtask bench layout` time ember's
//! kernels (Chapter 5); `cargo xtask bench threads` on threads (Chapter 24).

mod bench;
mod check;

use std::process::{Command, ExitCode};

// @listing main
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // A slice pattern matches `&str`s, so match on a Vec<&str> of the arguments.
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        // @hide
        ["check", ref rest @ ..] => check::main(rest),
        // @show
        ["bench", ..] if cfg!(debug_assertions) => in_release(&args),
        ["bench", "matmul", ref sizes @ ..] => {
            let sizes: Result<Vec<usize>, _> =
                sizes.iter().map(|s| s.parse()).collect();
            match sizes {
                Ok(s) if s.is_empty() => bench::matmul(&[256, 512]),
                Ok(s) => bench::matmul(&s),
                Err(_) => return usage(),
            }
            ExitCode::SUCCESS
        }
        // @hide
        ["bench", "layout"] => {
            bench::layout();
            ExitCode::SUCCESS
        }
        ["bench", "nt", n] => with_size(n, bench::nt),
        ["bench", "panel", n] => with_size(n, bench::panel),
        ["bench", "threads"] => {
            bench::threads(1024);
            bench::spawn();
            ExitCode::SUCCESS
        }
        // @show
        _ => usage(),
    }
}

fn usage() -> ExitCode {
    eprintln!("usage:");
    // @hide
    eprintln!(
        "  cargo xtask check <chapter | chapter.challenge | all> [--crate <dir>]"
    );
    // @show
    eprintln!("  cargo xtask bench matmul [sizes…]");
    // @hide
    eprintln!("  cargo xtask bench layout");
    eprintln!("  cargo xtask bench nt n");
    eprintln!("  cargo xtask bench panel n");
    eprintln!("  cargo xtask bench threads");
    // @show
    ExitCode::from(2)
}
// @end

// @listing with-size
/// Runs `bench(n)` for the size the argument `n` names (Challenges 5.4 and 5.6).
fn with_size(n: &str, bench: fn(usize)) -> ExitCode {
    match n.parse() {
        Ok(n) => {
            bench(n);
            ExitCode::SUCCESS
        }
        Err(_) => usage(),
    }
}
// @end

// @listing in-release
/// Timings from a debug build measure the debug build. So `bench` in a debug build
/// runs itself again, compiled with optimizations, and passes its arguments on.
/// (The release profile must keep debug assertions off, or this never stops.)
fn in_release(args: &[String]) -> ExitCode {
    let status = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--release", "--package", "xtask", "--"])
        .args(args)
        .status();
    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => panic!("bench: can't run {}: {e}", env!("CARGO")),
    }
}
// @end
