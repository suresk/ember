//! `cargo xtask check 2` runs Chapter 2's tests (the chapter's examples and its
//! challenges) and says which ones still fail.
//! `cargo xtask check all` runs every chapter's tests.
//! `cargo xtask bench matmul [sizes]` and `cargo xtask bench layout` time ember's
//! kernels (Chapter 5); `cargo xtask bench threads` on threads (Chapter 24).

mod bench;

use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["check", "all"] => run(&["test", "--package", "ember"], "all chapters"),
        ["check", n] => match n.parse::<u32>() {
            Ok(n) => {
                let tests = chapter_tests(n);
                if tests.is_empty() {
                    eprintln!("no tests for Chapter {n} in ember/tests");
                    return ExitCode::from(2);
                }
                let mut args = vec!["test", "--package", "ember"];
                for t in &tests {
                    args.extend(["--test", t.as_str()]);
                }
                run(&args, &format!("Chapter {n}"))
            }
            Err(_) => usage(),
        },
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
        ["bench", "layout"] => {
            bench::layout();
            ExitCode::SUCCESS
        }
        ["bench", "threads"] => {
            bench::threads(1024);
            bench::spawn();
            ExitCode::SUCCESS
        }
        ["bench", which @ ("panel" | "nt"), n] => match n.parse() {
            Ok(n) if which == "panel" => {
                bench::panel(n);
                ExitCode::SUCCESS
            }
            Ok(n) => {
                bench::nt(n);
                ExitCode::SUCCESS
            }
            Err(_) => usage(),
        },
        _ => usage(),
    }
}

/// The test files for Chapter `n`: `book_chNN` (the examples the chapter prints),
/// `challenges_chNN` (its challenges), and any `_alloc` variants of the two.
fn chapter_tests(n: u32) -> Vec<String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../ember/tests");
    let (book, challenges) =
        (format!("book_ch{n:02}"), format!("challenges_ch{n:02}"));
    let mut tests: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter_map(|name| name.strip_suffix(".rs").map(str::to_string))
        .filter(|stem| {
            [&book, &challenges].iter().any(|prefix| {
                stem.strip_prefix(prefix.as_str())
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
        })
        .collect();
    tests.sort();
    tests
}

// @listing in-release
/// Timings from a debug build measure the debug build. So `bench` in a debug build
/// runs itself again, compiled with optimizations, and passes its arguments on.
fn in_release(args: &[String]) -> ExitCode {
    let status = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--release", "--package", "xtask", "--"])
        .args(args)
        .status();
    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}
// @end

fn usage() -> ExitCode {
    eprintln!("usage: cargo xtask check <chapter number | all>");
    eprintln!("       cargo xtask bench <matmul [sizes…] | layout | panel n | nt n>");
    eprintln!("       cargo xtask bench threads");
    ExitCode::from(2)
}

fn run(args: &[&str], what: &str) -> ExitCode {
    let out = Command::new(env!("CARGO"))
        .args(args)
        .args(["--", "--test-threads=1"])
        .output()
        .expect("could not run cargo");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.contains("Running") && !out.status.success() {
        // It didn't compile: show the compiler's errors as they are.
        eprint!("{stderr}");
        return ExitCode::FAILURE;
    }
    let (mut passed, mut failed, mut todo) = (vec![], vec![], vec![]);
    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("test ") else {
            continue;
        };
        if let Some(name) = rest.strip_suffix(" ... ok") {
            passed.push(name.to_string());
        } else if let Some(name) = rest.strip_suffix(" ... FAILED") {
            let name = name.trim_end_matches(" - should panic");
            // A todo!() you haven't replaced yet panics with "not yet implemented".
            let section = format!("---- {name} stdout ----");
            let msg = stdout.split(&section).nth(1).unwrap_or("");
            if msg.contains("not yet implemented") {
                todo.push(name.to_string());
            } else {
                failed.push((name.to_string(), first_panic(msg)));
            }
        }
    }
    println!(
        "{what}: {} passed, {} failed, {} not written yet",
        passed.len(),
        failed.len(),
        todo.len()
    );
    for (name, why) in &failed {
        println!("  ✗ {name}\n      {why}");
    }
    for name in &todo {
        println!("  · {name} (still todo!())");
    }
    if failed.is_empty() && todo.is_empty() {
        println!("All done.");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// The panic message of a failed test: the line after "panicked at …".
fn first_panic(msg: &str) -> String {
    let mut lines = msg.lines().skip_while(|l| !l.contains("panicked at"));
    lines.next();
    lines
        .take_while(|l| !l.trim().is_empty() && !l.starts_with("note:"))
        .collect::<Vec<_>>()
        .join(" ")
}
