//! Chapter 14: pictures in the terminal. A loss curve on a log scale, and a map of
//! the classes a model predicts over the plane. Chapter 18 adds a curve on a
//! linear scale, for learning rates that start or end at zero.

use crate::math;
use crate::tensor::Tensor;

// @listing curve
/// A text chart of `values`, `width` columns by `height` rows, on a log scale:
/// column j shows the mean of its share of the values.
pub fn curve(values: &[f32], width: usize, height: usize) -> String {
    let n = values.len();
    let column = |j: usize| {
        let share = &values[j * n / width..(j + 1) * n / width];
        math::ln(share.iter().sum::<f32>() / share.len() as f32)
    };
    let logs: Vec<f32> = (0..width).map(column).collect();
    let span = |(t, b): (f32, f32), &y: &f32| (t.max(y), b.min(y));
    let (top, bottom) = logs.iter().fold((f32::MIN, f32::MAX), span);
    let mut lines = vec![vec![' '; width]; height];
    for (j, &y) in logs.iter().enumerate() {
        let row = (top - y) / (top - bottom).max(1e-6) * (height - 1) as f32;
        lines[row.round() as usize][j] = '*';
    }
    // @end
    // @listing curve-labels
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let label = match i {
            0 => format!("{:>8.4}", math::exp(top)),
            i if i == height - 1 => format!("{:>8.4}", math::exp(bottom)),
            _ => " ".repeat(8),
        };
        out += &format!("{label} |{}\n", line.iter().collect::<String>());
    }
    out += &format!("{:>9}+{}\n{:>10}{:>width$}\n", "", "-".repeat(width), 0, n);
    out
}
// @end

/// `curve` on a linear scale: column j shows the mean of its share of the values,
/// between their smallest and largest column means.
pub fn curve_linear(values: &[f32], width: usize, height: usize) -> String {
    let n = values.len();
    let column = |j: usize| {
        let share = &values[j * n / width..(j + 1) * n / width];
        share.iter().sum::<f32>() / share.len() as f32
    };
    let ys: Vec<f32> = (0..width).map(column).collect();
    let top = ys.iter().cloned().fold(f32::MIN, f32::max);
    let bottom = ys.iter().cloned().fold(f32::MAX, f32::min);
    let mut lines = vec![vec![' '; width]; height];
    for (j, &y) in ys.iter().enumerate() {
        let row = (top - y) / (top - bottom).max(1e-12) * (height - 1) as f32;
        lines[row.round() as usize][j] = '*';
    }
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let label = match i {
            0 => format!("{top:>8.4}"),
            i if i == height - 1 => format!("{bottom:>8.4}"),
            _ => " ".repeat(8),
        };
        out += &format!("{label} |{}\n", line.iter().collect::<String>());
    }
    out += &format!("{:>9}+{}\n{:>10}{:>width$}\n", "", "-".repeat(width), 0, n);
    out
}

// @listing regions
/// A map of the square [-extent, extent]², `cols` characters by `rows`: each
/// character shows the class `classify` gives the center of its cell, `.` for 0,
/// a space for 1, then `-`, `:`; the `points` are drawn on top as `o`, `x`, `+`, `#`.
pub fn regions(
    classify: impl Fn(&Tensor) -> Vec<usize>,
    extent: f32,
    (cols, rows): (usize, usize),
    (points, labels): (&Tensor, &[usize]),
) -> String {
    let at = |k: usize, n: usize| extent * (2.0 * (k as f32 + 0.5) / n as f32 - 1.0);
    let mut centers = Vec::with_capacity(2 * rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            centers.extend([at(c, cols), -at(r, rows)]); // the top row first
        }
    }
    let classes = classify(&Tensor::from_vec(centers, &[rows * cols, 2]));
    let mut grid: Vec<Vec<char>> = classes
        .chunks(cols)
        .map(|line| line.iter().map(|&c| ['.', ' ', '-', ':'][c]).collect())
        .collect();
    for (i, &label) in labels.iter().enumerate() {
        let cell = |v: f32, n: usize| ((v / extent + 1.0) / 2.0 * n as f32).floor();
        let (c, r) = (
            cell(points.get(&[i, 0]), cols),
            cell(-points.get(&[i, 1]), rows),
        );
        if (0.0..cols as f32).contains(&c) && (0.0..rows as f32).contains(&r) {
            grid[r as usize][c as usize] = ['o', 'x', '+', '#'][label];
        }
    }
    grid.iter()
        .map(|line| line.iter().collect::<String>() + "\n")
        .collect()
}
// @end
