//! `cargo xtask check`: runs a chapter's tests and reports on the chapter's examples
//! and on each of its challenges separately.
//!
//! The tests are the ones in `ember/tests`. By default they run against the crate in
//! this repository. With `--crate <dir>` (or the `EMBER_CRATE` environment variable)
//! they run against the crate in that directory, which is how you check your own
//! code against them without copying anything into it. The directory can be the
//! crate's, or a workspace that has the crate in `ember/`, as the book sets it up.
//!
//! How it works: a small package is written to `target/check`. Its only dependency
//! is the crate being checked, under the name `ember`, and its `tests/` directory
//! holds copies of the test files. A challenge file is copied once per challenge,
//! each copy keeping only that challenge's tests, so a challenge you haven't written
//! yet cannot stop the others from compiling. The lines that are left out are
//! replaced by empty lines, so line numbers in messages match the original file.
//!
//! A test that includes a file from `examples/` gets the one in the crate being
//! checked.
//!
//! A challenge's tests start at a `// @challenge 3` line and run to the next such
//! line or the end of the file. Lines that two challenges need, such as an import
//! both use, go after a `// @challenge 2 5` line. Everything before the first marker
//! (imports and helpers) is shared by all of them, so it must use nothing a
//! challenge asks for.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Output};
use std::{env, fs, io};

/// The root of this repository.
const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

/// What the command line asked for.
#[derive(Clone, Copy)]
enum Which {
    All,
    Chapter(u32),
    Challenge(u32, u32),
}

/// What one test binary checks.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    /// The chapter's examples, from `book_chNN.rs`.
    Examples,
    /// One challenge.
    Challenge(u32),
    /// Every challenge at once, from a file that has no `// @challenge` lines.
    Challenges,
}

/// One test binary in the scratch package.
struct Target {
    chapter: u32,
    part: Part,
    /// Its name there: `tests/<name>.rs`.
    name: String,
    /// The file in `ember/tests` it was made from, without `.rs`.
    stem: String,
    source: String,
}

/// How one test binary did.
enum Outcome {
    Ran {
        passed: usize,
        /// Name and panic message.
        failed: Vec<(String, String)>,
        /// Tests that reached a `todo!()`.
        todo: Vec<String>,
        output: String,
    },
    /// It doesn't compile, and these names it uses don't exist yet.
    Missing { names: Vec<String>, output: String },
    /// It doesn't compile for another reason.
    Broken { summary: String, output: String },
}

pub fn main(args: &[&str]) -> ExitCode {
    let (which, dir) = match parse(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match check(which, dir) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}

const USAGE: &str = "usage: cargo xtask check <chapter | chapter.challenge | all> \
                     [--crate <dir>]";

fn parse(args: &[&str]) -> Result<(Which, Option<PathBuf>), String> {
    let (what, dir) = match *args {
        [what] => (what, env::var_os("EMBER_CRATE").map(PathBuf::from)),
        [what, "--crate", dir] | ["--crate", dir, what] => (what, Some(dir.into())),
        _ => return Err(USAGE.to_string()),
    };
    let number = |s: &str| s.parse::<u32>().map_err(|_| USAGE.to_string());
    let which = match what.split_once('.') {
        _ if what == "all" => Which::All,
        Some((chapter, challenge)) => {
            Which::Challenge(number(chapter)?, number(challenge)?)
        }
        None => Which::Chapter(number(what)?),
    };
    Ok((which, dir))
}

/// Runs the tests. `Ok(true)` means every one of them passed.
fn check(which: Which, dir: Option<PathBuf>) -> Result<bool, String> {
    let targets = select(which)?;
    let reference = dir.is_none();
    let dir = dir.unwrap_or_else(|| Path::new(ROOT).join("ember"));
    let dir =
        std::path::absolute(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    // A workspace like the book's, `ember-workspace/` with the crate in `ember/`, will do:
    // its Cargo.toml has no [package], so look for the crate inside it.
    let dir =
        match dir.join("ember/Cargo.toml").is_file() && package_name(&dir).is_err() {
            true => dir.join("ember"),
            false => dir,
        };
    let package = package_name(&dir)?;
    let scratch = Path::new(ROOT).join("target").join("check");
    write_package(&scratch, &dir, &package, &targets)
        .map_err(|e| format!("could not write {}: {e}", scratch.display()))?;

    if reference {
        println!("Checking the reference crate (--crate <dir> checks yours).\n");
    } else {
        // Without the `..`s of a relative path. Only for the message: Cargo is given
        // `dir` itself, because on Windows a canonical path has a prefix it dislikes.
        let shown = fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        println!("Checking {}.\n", shown.display());
    }

    // Build every test binary that can be built, in parallel. The library is built
    // first, and if that fails there is nothing to test.
    let mut build = cargo(&scratch);
    build.args(["build", "--keep-going"]);
    for target in &targets {
        build.args(["--test", &target.name]);
    }
    let built = run(&mut build)?;
    let stderr = String::from_utf8_lossy(&built.stderr);
    if stderr.contains(&format!("could not compile `{package}` (lib)")) {
        eprint!("{stderr}");
        return Err(format!("\n{package} does not compile, so no test can run."));
    }

    let full = matches!(which, Which::Challenge(..));
    let mut all_passed = true;
    let mut chapter = 0;
    for target in &targets {
        if target.chapter != chapter && !full {
            chapter = target.chapter;
            println!("Chapter {chapter}");
        }
        let outcome = run_target(&scratch, target)?;
        all_passed &= report(target, &outcome, full);
    }
    if !all_passed
        && !full
        && targets.iter().any(|t| matches!(t.part, Part::Challenge(_)))
    {
        println!("\n`cargo xtask check <chapter>.<challenge>` shows one in full.");
    }
    Ok(all_passed)
}

/// The test binaries for what was asked.
fn select(which: Which) -> Result<Vec<Target>, String> {
    match which {
        Which::All => Ok(chapters().into_iter().flat_map(targets).collect()),
        Which::Chapter(n) => match targets(n) {
            found if found.is_empty() => Err(format!("no tests for Chapter {n}")),
            found => Ok(found),
        },
        Which::Challenge(n, k) => {
            let (mut found, mut split) = (vec![], BTreeSet::new());
            for target in targets(n) {
                match target.part {
                    Part::Challenge(c) if c == k => found.push(target),
                    Part::Challenge(c) => drop(split.insert(c)),
                    _ => {}
                }
            }
            if !found.is_empty() {
                Ok(found)
            } else if split.is_empty() {
                Err(format!(
                    "Chapter {n}'s challenge tests aren't split by challenge. \
                     `cargo xtask check {n}` runs them together."
                ))
            } else {
                let have: Vec<String> =
                    split.iter().map(|c| format!("{n}.{c}")).collect();
                Err(format!(
                    "no tests for Challenge {n}.{k}. There are: {}",
                    have.join(", ")
                ))
            }
        }
    }
}

/// The names of the files in `ember/tests`, without `.rs`, in order.
fn test_files() -> Vec<String> {
    let mut stems: Vec<String> = fs::read_dir(Path::new(ROOT).join("ember/tests"))
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter_map(|name| name.strip_suffix(".rs").map(str::to_string))
        .collect();
    stems.sort();
    stems
}

/// Splits `book_ch07` or `challenges_ch15_alloc` into its kind and chapter number.
fn kind_and_chapter(stem: &str) -> Option<(&str, u32)> {
    let (kind, rest) = stem.split_once("_ch")?;
    let digits = rest.split('_').next()?;
    matches!(kind, "book" | "challenges").then_some((kind, digits.parse().ok()?))
}

/// Every chapter that has tests.
fn chapters() -> Vec<u32> {
    let all: BTreeSet<u32> = test_files()
        .iter()
        .filter_map(|stem| Some(kind_and_chapter(stem)?.1))
        .collect();
    all.into_iter().collect()
}

/// The test binaries for Chapter `n`: its examples first, then its challenges.
fn targets(n: u32) -> Vec<Target> {
    let mut targets = vec![];
    for stem in test_files() {
        let Some((kind, chapter)) = kind_and_chapter(&stem) else {
            continue;
        };
        if chapter != n {
            continue;
        }
        let path = Path::new(ROOT)
            .join("ember/tests")
            .join(format!("{stem}.rs"));
        let Ok(source) = fs::read_to_string(path) else {
            continue;
        };
        let whole = |part, source| Target {
            chapter,
            part,
            name: stem.clone(),
            stem: stem.clone(),
            source,
        };
        let slices = split(&source);
        if kind == "book" {
            targets.push(whole(Part::Examples, source));
        } else if slices.is_empty() {
            targets.push(whole(Part::Challenges, source));
        } else {
            for (k, source) in slices {
                targets.push(Target {
                    chapter,
                    part: Part::Challenge(k),
                    name: format!("{stem}_c{k}"),
                    stem: stem.clone(),
                    source,
                });
            }
        }
    }
    targets.sort_by_key(|t| t.part);
    targets
}

/// The numbers on a `// @challenge 3` line, or a `// @challenge 2 5` line for lines
/// that two challenges share.
fn marker(line: &str) -> Option<Vec<u32>> {
    let numbers = line
        .trim()
        .strip_prefix("// @challenge ")?
        .split_whitespace();
    numbers.map(|n| n.parse().ok()).collect()
}

/// One copy of a challenge file per challenge. Each keeps the lines before the first
/// `// @challenge` line and the lines of its own challenge. The other lines are
/// emptied, not removed, so every line keeps its number.
fn split(source: &str) -> BTreeMap<u32, String> {
    let mut owners: Vec<Option<Vec<u32>>> = vec![];
    let mut owner = None;
    for line in source.lines() {
        owner = marker(line).or(owner);
        owners.push(owner.clone());
    }
    let challenges: BTreeSet<u32> =
        owners.iter().flatten().flatten().copied().collect();
    let slice = |k| {
        // A slice leaves some imports and helpers unused. The attribute shares line 1
        // with whatever was there.
        let mut out = String::from("#![allow(unused)] ");
        for (line, owner) in source.lines().zip(&owners) {
            if owner.as_ref().is_none_or(|o| o.contains(&k)) {
                out.push_str(line);
            }
            out.push('\n');
        }
        out
    };
    challenges.into_iter().map(|k| (k, slice(k))).collect()
}

/// The `name` in the `[package]` section of the `Cargo.toml` in `dir`.
fn package_name(dir: &Path) -> Result<String, String> {
    let manifest = dir.join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .map_err(|e| format!("{}: {e}", manifest.display()))?;
    text.lines()
        .skip_while(|line| line.trim() != "[package]")
        .skip(1)
        .take_while(|line| !line.trim_start().starts_with('['))
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            let value = value.trim().trim_matches(['"', '\'']);
            (key.trim() == "name").then(|| value.to_string())
        })
        .ok_or(format!("{}: no [package] with a name", manifest.display()))
}

/// Writes the scratch package: a manifest, an empty library and the test files.
/// A file that already has the right contents is left alone, so Cargo doesn't
/// rebuild it.
fn write_package(
    scratch: &Path,
    dir: &Path,
    package: &str,
    targets: &[Target],
) -> io::Result<()> {
    fs::create_dir_all(scratch.join("src"))?;
    fs::create_dir_all(scratch.join("tests"))?;
    // The empty [workspace] keeps this package out of the workspace around it.
    let manifest = format!(
        "[package]\nname = \"check\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
         publish = false\n\n[dependencies]\n\
         ember = {{ path = '{}', package = \"{package}\" }}\n\n[workspace]\n",
        dir.display()
    );
    write_if_changed(&scratch.join("Cargo.toml"), &manifest)?;
    write_if_changed(&scratch.join("src/lib.rs"), "")?;
    // Some tests include a file from `examples/` with `#[path = "../examples/x.rs"]`.
    // In the copy that has to be the examples directory of the crate being checked,
    // written as a Rust string so that a Windows path keeps its backslashes.
    let examples = dir.join("examples").join("").display().to_string();
    let examples = format!("#[path = {:?}", examples);
    let examples = examples.trim_end_matches('"');
    for target in targets {
        let path = scratch.join("tests").join(format!("{}.rs", target.name));
        let source = target.source.replace("#[path = \"../examples/", examples);
        write_if_changed(&path, &source)?;
    }
    Ok(())
}

fn write_if_changed(path: &Path, contents: &str) -> io::Result<()> {
    match fs::read_to_string(path) {
        Ok(old) if old == contents => Ok(()),
        _ => fs::write(path, contents),
    }
}

fn cargo(scratch: &Path) -> Command {
    let mut cargo = Command::new(env!("CARGO"));
    cargo.current_dir(scratch);
    cargo
}

fn run(command: &mut Command) -> Result<Output, String> {
    command
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))
}

/// Builds and runs one test binary.
fn run_target(scratch: &Path, target: &Target) -> Result<Outcome, String> {
    let mut test = cargo(scratch);
    test.args(["test", "--test", &target.name, "--", "--test-threads=1"]);
    let out = run(&mut test)?;
    // Messages name the copy. Name the file it was copied from instead.
    let copy = format!("tests/{}.rs", target.name);
    let original = format!("ember/tests/{}.rs", target.stem);
    let text =
        |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&copy, &original);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    if !out.status.success() && !stderr.contains("Running") {
        return Ok(compile_failure(stderr, &target.source));
    }

    let (mut passed, mut failed, mut todo) = (0, vec![], vec![]);
    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("test ") else {
            continue;
        };
        if rest.ends_with(" ... ok") {
            passed += 1;
        } else if let Some(name) = rest.strip_suffix(" ... FAILED") {
            let name = name.trim_end_matches(" - should panic");
            // A todo!() that hasn't been replaced panics with "not yet implemented".
            let section = format!("---- {name} stdout ----");
            let message = stdout.split(&section).nth(1).unwrap_or("");
            if message.contains("not yet implemented") {
                todo.push(name.to_string());
            } else {
                failed.push((name.to_string(), first_panic(message)));
            }
        }
    }
    let total = passed + failed.len() + todo.len();
    if out.status.success() == (failed.is_empty() && todo.is_empty()) && total > 0 {
        return Ok(Outcome::Ran {
            passed,
            failed,
            todo,
            output: stdout,
        });
    }
    // Something else went wrong (a crash, or no tests at all): show everything.
    let summary = "the tests did not run to the end".to_string();
    Ok(Outcome::Broken {
        summary,
        output: stdout + &stderr,
    })
}

/// Sorts a compiler failure into "these names don't exist yet" and everything else.
/// `source` is the test file as compiled, for finding the name an error points at.
fn compile_failure(stderr: String, source: &str) -> Outcome {
    // Leave out the warnings about the crate being checked, which come first.
    let start = stderr.find("\nerror").map_or(0, |at| at + 1);
    let stderr = stderr[start..].to_string();
    // A missing name often causes other errors after it, so one is enough.
    let mut missing = false;
    let mut names = vec![];
    for error in errors(&stderr) {
        if let Some(name) = missing_name(&error, source) {
            missing = true;
            names.extend(name);
        }
    }
    if missing {
        names.sort();
        names.dedup();
        return Outcome::Missing {
            names,
            output: stderr,
        };
    }
    // The first error and where it is.
    let mut lines = stderr.lines().skip_while(|line| !line.starts_with("error"));
    let what = lines.next().unwrap_or("it does not compile");
    let at = lines.find(|line| line.trim_start().starts_with("-->"));
    let summary = match at {
        Some(at) => format!("{what}\n        {}", at.trim()),
        None => what.to_string(),
    };
    Outcome::Broken {
        summary,
        output: stderr,
    }
}

/// The compiler's errors, each as its lines: `error[E0599]: …` and what follows it,
/// up to the next error or warning.
fn errors(stderr: &str) -> Vec<Vec<&str>> {
    let mut errors: Vec<Vec<&str>> = vec![];
    let mut inside = false;
    for line in stderr.lines() {
        if line.starts_with("error[") {
            errors.push(vec![line]);
            inside = true;
        } else if line.starts_with("error") || line.starts_with("warning") {
            inside = false;
        } else if inside && let Some(error) = errors.last_mut() {
            error.push(line);
        }
    }
    errors
}

/// `Some` when `error` says something the test uses doesn't exist yet: a function,
/// method, type, trait or import that isn't found, or a trait the test needs that a
/// type of the crate doesn't implement. Inside is the name, when it can be found.
fn missing_name(error: &[&str], source: &str) -> Option<Option<String>> {
    // Unresolved imports, and functions, methods, types and traits that aren't found.
    const NOT_FOUND: [&str; 6] =
        ["E0432", "E0425", "E0599", "E0433", "E0412", "E0405"];
    let first = error.first()?;
    let code = first.strip_prefix("error[")?.split(']').next()?;
    let quoted: Vec<&str> = first.split('`').skip(1).step_by(2).collect();
    let last = |path: &str| path.rsplit("::").next().unwrap_or(path).to_string();
    // "`Tensor` doesn't implement `std::fmt::Display`" (E0277, or E0599 for a method
    // such as `to_string` that the trait would give), and "the trait bound
    // `Tensor: Display` is not satisfied".
    let implement = |ty: &str, tr: &str| {
        let ty = ty.trim();
        let ours = ty.starts_with(|c: char| c.is_ascii_uppercase());
        let tr = last(tr.split('<').next().unwrap_or(tr).trim());
        ours.then(|| format!("impl {tr} for {}", last(ty)))
    };
    if first.contains("doesn't implement")
        && quoted.len() >= 2
        && let Some(name) = implement(quoted[0], quoted[1])
    {
        return Some(Some(name));
    }
    if first.contains("the trait bound") && first.contains("is not satisfied") {
        let (ty, tr) = quoted.first()?.split_once(": ")?;
        return implement(ty, tr).map(Some);
    }
    if !NOT_FOUND.contains(&code) {
        return None;
    }
    if first.contains("is not an iterator") {
        // "`Gradients` is not an iterator" names the type: the missing name is the
        // method under the arrow, as in `grads.count()`.
        return Some(pointed_at(error, source));
    }
    Some(quoted.first().map(|path| last(path)))
}

/// The identifier at the `--> file:line:column` an error points at.
fn pointed_at(error: &[&str], source: &str) -> Option<String> {
    let at = error
        .iter()
        .find_map(|l| l.trim_start().strip_prefix("--> "))?;
    let mut parts = at.rsplitn(3, ':');
    let column: usize = parts.next()?.trim().parse().ok()?;
    let line: usize = parts.next()?.parse().ok()?;
    let text = source.lines().nth(line.checked_sub(1)?)?;
    let rest = text.chars().skip(column.checked_sub(1)?);
    let name: String = rest
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Prints how one test binary did. Returns whether all of it passed.
fn report(target: &Target, outcome: &Outcome, full: bool) -> bool {
    let label = match target.part {
        Part::Examples => "Examples".to_string(),
        Part::Challenge(k) => format!("Challenge {}.{k}", target.chapter),
        Part::Challenges => "Challenges".to_string(),
    };
    // A second file for the same part (`…_alloc.rs`) says which file it is.
    let label = match target.stem.rsplit_once('_') {
        Some((_, suffix)) if !suffix.starts_with("ch") => {
            format!("{label} ({suffix})")
        }
        _ => label,
    };
    let line = |status: &str| println!("  {label:<17} {status}");
    let tests = |n: usize| {
        if n == 1 {
            "1 test".to_string()
        } else {
            format!("{n} tests")
        }
    };
    let output = match outcome {
        Outcome::Ran {
            passed,
            failed,
            todo,
            output,
        } => {
            if failed.is_empty() && todo.is_empty() {
                line(&format!("passed ({})", tests(*passed)));
                return true;
            }
            let total = passed + failed.len() + todo.len();
            if failed.is_empty() {
                line("not written yet (it reaches a todo!())");
            } else {
                line(&format!("FAILED ({} of {})", failed.len(), tests(total)));
            }
            for (name, why) in failed {
                println!("      ✗ {name}\n        {why}");
            }
            for name in todo {
                println!("      · {name} (still todo!())");
            }
            output
        }
        Outcome::Missing { names, output } => {
            if names.is_empty() {
                line("not started (it uses a name that doesn't exist yet)");
            } else {
                let names: Vec<String> =
                    names.iter().map(|n| format!("`{n}`")).collect();
                line(&format!("not started (no {})", names.join(", ")));
            }
            output
        }
        Outcome::Broken { summary, output } => {
            line("does not compile");
            println!("      {summary}");
            output
        }
    };
    if full {
        println!("\n{}", output.trim_end());
    }
    false
}

/// The panic message of a failed test: the lines after "panicked at …".
fn first_panic(message: &str) -> String {
    let mut lines = message.lines().skip_while(|l| !l.contains("panicked at"));
    lines.next();
    lines
        .take_while(|l| !l.trim().is_empty() && !l.starts_with("note:"))
        .collect::<Vec<_>>()
        .join(" ")
}
