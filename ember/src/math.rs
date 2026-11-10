//! `exp`, `ln`, `tanh`, `sin`, `cos` and powers, the same on every machine.
//!
//! The standard library's versions of these functions call the platform's math
//! library, and the IEEE standard doesn't require those to round correctly, so
//! `1.5f32.exp()` can differ in its last bit between macOS, Linux and Windows. One
//! bit is harmless in a single result. Training is not a single result: a few
//! hundred steps amplify the bit until a loss differs in its second decimal. Then
//! the numbers in the book would hold on one platform only.
//!
//! The functions here use only `+`, `-`, `*`, `/` and `sqrt`, which IEEE 754 does
//! define exactly, so they return the same bits everywhere. Each one computes in
//! `f64` and rounds once to `f32`. That makes the `f32` result correct to within
//! a hair of half a unit in the last place, which is as good as the platform
//! libraries and far better than ember needs.
//!
//! The method is the classical one for each function: reduce the argument to a
//! small interval around zero, where a short polynomial is accurate to `f64`
//! precision, then undo the reduction.

const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000); // ln 2, top 33 bits
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76); // and the rest of it
const INV_LN2: f64 = std::f64::consts::LOG2_E;

// π/2 in three pieces. The first two have trailing zero bits, so that `k * piece`
// is exact for the multiples `k` that ember's angles reach.
const PIO2_1: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
const PIO2_2: f64 = f64::from_bits(0x3dd0_b461_1a60_0000);
const PIO2_3: f64 = f64::from_bits(0x3ba3_198a_2e03_7073);

// The series' coefficients. The compiler evaluates these divisions, by the same
// IEEE rules on every host, so each is the correctly rounded f64.
/// 1/n! for n = 0 to 13: the Taylor series of e^r.
const EXP_SERIES: [f64; 14] = [
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
];
/// 1/(2n + 1) for n = 0 to 12: the series of ln((1 + s) / (1 - s)) / 2s.
const ODD_RECIPROCALS: [f64; 13] = [
    1.0,
    1.0 / 3.0,
    1.0 / 5.0,
    1.0 / 7.0,
    1.0 / 9.0,
    1.0 / 11.0,
    1.0 / 13.0,
    1.0 / 15.0,
    1.0 / 17.0,
    1.0 / 19.0,
    1.0 / 21.0,
    1.0 / 23.0,
    1.0 / 25.0,
];
/// ±1/(2n + 1)! for n = 0 to 9: the Taylor series of sin r / r, in powers of r².
const SIN_SERIES: [f64; 10] = [
    1.0,
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5040.0,
    1.0 / 362880.0,
    -1.0 / 39916800.0,
    1.0 / 6227020800.0,
    -1.0 / 1307674368000.0,
    1.0 / 355687428096000.0,
    -1.0 / 121645100408832000.0,
];
/// ±1/(2n)! for n = 0 to 10: the Taylor series of cos r, in powers of r².
const COS_SERIES: [f64; 11] = [
    1.0,
    -1.0 / 2.0,
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40320.0,
    -1.0 / 3628800.0,
    1.0 / 479001600.0,
    -1.0 / 87178291200.0,
    1.0 / 20922789888000.0,
    -1.0 / 6402373705728000.0,
    1.0 / 2432902008176640000.0,
];

/// `x · 2^k`, built from the bits of the power of two.
fn scale(x: f64, k: i32) -> f64 {
    let two_to = |k: i32| f64::from_bits(((k + 1023) as u64) << 52);
    match k {
        1024.. => x * two_to(1023) * two_to((k - 1023).min(1023)),
        -1022..=1023 => x * two_to(k),
        _ => x * two_to((k + 1000).max(-1022)) * two_to(-1000), // toward zero
    }
}

/// e^x for an `f64`.
pub fn exp64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.79 {
        return f64::INFINITY;
    }
    if x < -745.14 {
        return 0.0;
    }
    // x = k ln 2 + r with |r| ≤ ½ ln 2, so e^x = 2^k e^r.
    let k = (x * INV_LN2).round();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    // The Taylor series of e^r to r^13 / 13!, by Horner's rule.
    let mut sum = 0.0;
    for c in EXP_SERIES.iter().rev() {
        sum = sum * r + c;
    }
    scale(sum, k as i32)
}

/// The natural logarithm of an `f64`.
pub fn ln64(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }
    // x = m · 2^e with m in [√½, √2), so ln x = e ln 2 + ln m.
    let (x, lift) = if x < f64::MIN_POSITIVE {
        (x * scale(1.0, 54), -54) // a subnormal: make it normal first
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023 + lift;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023 << 52));
    if m > std::f64::consts::SQRT_2 {
        m /= 2.0;
        e += 1;
    }
    // ln m = 2 (s + s³/3 + s⁵/5 + …) for s = (m - 1) / (m + 1), and |s| < 0.172.
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut sum = 0.0;
    for c in ODD_RECIPROCALS.iter().rev() {
        sum = sum * s2 + c;
    }
    e as f64 * LN2_HI + (2.0 * s * sum + e as f64 * LN2_LO)
}

/// `x` minus the nearest multiple of π/2, and which multiple that was (mod 4).
fn reduce(x: f64) -> (f64, i64) {
    let k = (x * std::f64::consts::FRAC_2_PI).round();
    let r = ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
    (r, (k as i64).rem_euclid(4))
}

/// sin r for |r| ≤ π/4: the Taylor series to r^19 / 19!.
fn sin_kernel(r: f64) -> f64 {
    let r2 = r * r;
    let mut sum = 0.0;
    for c in SIN_SERIES.iter().rev() {
        sum = sum * r2 + c;
    }
    r * sum
}

/// cos r for |r| ≤ π/4: the Taylor series to r^20 / 20!.
fn cos_kernel(r: f64) -> f64 {
    let r2 = r * r;
    let mut sum = 0.0;
    for c in COS_SERIES.iter().rev() {
        sum = sum * r2 + c;
    }
    sum
}

/// The sine of an `f64` (in radians).
pub fn sin64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    match reduce(x) {
        (r, 0) => sin_kernel(r),
        (r, 1) => cos_kernel(r),
        (r, 2) => -sin_kernel(r),
        (r, _) => -cos_kernel(r),
    }
}

/// The cosine of an `f64` (in radians).
pub fn cos64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    match reduce(x) {
        (r, 0) => cos_kernel(r),
        (r, 1) => -sin_kernel(r),
        (r, 2) => -cos_kernel(r),
        (r, _) => sin_kernel(r),
    }
}

/// The hyperbolic tangent of an `f64`.
pub fn tanh64(x: f64) -> f64 {
    let a = x.abs();
    let t = if a < 1e-4 {
        a - a * a * a / 3.0 // (e^2a - 1) would lose digits here; x⁵ is below f64
    } else if a > 20.0 {
        1.0
    } else {
        let e = exp64(2.0 * a);
        (e - 1.0) / (e + 1.0)
    };
    if x < 0.0 { -t } else { t } // keeps NaN a NaN
}

/// `x` to a whole-number power, by repeated squaring.
pub fn powi64(x: f64, n: i32) -> f64 {
    let (mut base, mut left, mut result) = (x, n.unsigned_abs(), 1.0);
    while left > 0 {
        if left & 1 == 1 {
            result *= base;
        }
        base *= base;
        left >>= 1;
    }
    if n < 0 { 1.0 / result } else { result }
}

/// `x` to the power `p`. Whole-number powers multiply, halves use `sqrt`, and
/// everything else is `exp(p ln x)`, which needs `x` to be positive.
pub fn powf64(x: f64, p: f64) -> f64 {
    if p == p.trunc() && p.abs() <= 1024.0 {
        powi64(x, p as i32)
    } else if p == 0.5 {
        x.sqrt()
    } else if p == -0.5 {
        1.0 / x.sqrt()
    } else if x == 0.0 {
        if p > 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        exp64(p * ln64(x)) // NaN for a negative x, as in the standard library
    }
}

// @listing f32
/// e^x.
pub fn exp(x: f32) -> f32 {
    exp64(x as f64) as f32
}

/// The natural logarithm.
pub fn ln(x: f32) -> f32 {
    ln64(x as f64) as f32
}

/// The hyperbolic tangent.
pub fn tanh(x: f32) -> f32 {
    tanh64(x as f64) as f32
}
// @end

/// ln(1 + x), accurate when `x` is small.
pub fn ln_1p(x: f32) -> f32 {
    let x = x as f64;
    let u = 1.0 + x;
    // The factor x / (u - 1) repairs what rounding 1 + x lost.
    (if u == 1.0 { x } else { ln64(u) * x / (u - 1.0) }) as f32
}

/// The sine (in radians).
pub fn sin(x: f32) -> f32 {
    sin64(x as f64) as f32
}

/// The cosine (in radians).
pub fn cos(x: f32) -> f32 {
    cos64(x as f64) as f32
}

/// `x` to the power `p`.
pub fn powf(x: f32, p: f32) -> f32 {
    powf64(x as f64, p as f64) as f32
}

/// `x` to a whole-number power.
pub fn powi(x: f32, n: i32) -> f32 {
    powi64(x as f64, n) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic sweep of `n` values across `[lo, hi]`.
    fn sweep(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..n).map(move |i| lo + (hi - lo) * (i as f32 + 0.37) / n as f32)
    }

    /// How many units in the last place apart two `f32`s of one sign are.
    fn ulps(a: f32, b: f32) -> u32 {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
    }

    #[test]
    fn within_one_ulp_of_the_standard_library() {
        // The platform's functions are themselves within an ulp of the truth, and
        // which side they land on is what differs from machine to machine.
        for x in sweep(-80.0, 80.0, 20_000) {
            assert!(ulps(exp(x), x.exp()) <= 1, "exp({x})");
            assert!(ulps(tanh(x / 8.0), (x / 8.0).tanh()) <= 2, "tanh({x} / 8)");
            // Near a zero of sin or cos an ulp is tiny, and the platforms differ more.
            let (s, c) = (sin(x), cos(x));
            assert!(
                ulps(s, x.sin()) <= 1 || (s - x.sin()).abs() < 1e-9,
                "sin({x})"
            );
            assert!(
                ulps(c, x.cos()) <= 1 || (c - x.cos()).abs() < 1e-9,
                "cos({x})"
            );
        }
        for x in sweep(1e-6, 50.0, 20_000) {
            for x in [x, x * 1e-30, x * 1e30] {
                assert!(ulps(ln(x), x.ln()) <= 1, "ln({x})");
            }
            assert!(ulps(powf(x, 1.7), x.powf(1.7)) <= 1, "{x}^1.7");
            assert!(ulps(powf(x, -0.3), x.powf(-0.3)) <= 1, "{x}^-0.3");
            assert!(
                ulps(ln_1p(x * 1e-5), (x * 1e-5).ln_1p()) <= 1,
                "ln_1p({x}e-5)"
            );
        }
    }

    #[test]
    fn f64_versions_agree_to_f64_precision() {
        for x in sweep(-30.0, 30.0, 5_000).map(f64::from) {
            assert!((exp64(x) / x.exp() - 1.0).abs() < 4e-16, "exp64({x})");
            assert!((sin64(x) - x.sin()).abs() < 4e-16, "sin64({x})");
            assert!((cos64(x) - x.cos()).abs() < 4e-16, "cos64({x})");
            assert!((tanh64(x) - x.tanh()).abs() < 4e-16, "tanh64({x})");
            let y = x.abs() + 1e-3;
            assert!((ln64(y) - y.ln()).abs() < 4e-16 * y.ln().abs().max(1.0));
        }
    }

    #[test]
    fn exact_cases_and_edges() {
        assert_eq!(
            (exp(0.0), ln(1.0), tanh(0.0), sin(0.0), cos(0.0)),
            (1.0, 0.0, 0.0, 0.0, 1.0)
        );
        assert_eq!(
            (powf(3.0, 2.0), powf(2.0, -1.0), powf(9.0, 0.5)),
            (9.0, 0.5, 3.0)
        );
        assert_eq!(
            (powf(-2.0, 3.0), powf(5.0, 0.0), powi(2.0, 10)),
            (-8.0, 1.0, 1024.0)
        );
        assert_eq!((exp(1000.0), exp(-1000.0)), (f32::INFINITY, 0.0));
        assert_eq!(
            (ln(0.0), tanh(50.0), tanh(-50.0)),
            (f32::NEG_INFINITY, 1.0, -1.0)
        );
        assert!(
            ln(-1.0).is_nan() && exp(f32::NAN).is_nan() && tanh(f32::NAN).is_nan()
        );
        assert!(sin(f32::INFINITY).is_nan() && powf(-2.0, 0.3).is_nan());
        assert_eq!(exp64(-745.0), 5e-324); // the smallest positive f64
        assert_eq!(ln64(5e-324), -744.4400719213812);
    }

    #[test]
    fn the_same_bits_on_every_machine() {
        // These were computed once and are asserted bit for bit. The standard
        // library's functions would pass this test on some platforms only.
        let mut hash = 0xcbf2_9ce4_8422_2325u64; // FNV-1a over the results' bits
        let mut mix = |y: f32| {
            for b in y.to_bits().to_le_bytes() {
                hash = (hash ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        };
        for x in sweep(-20.0, 20.0, 4_000) {
            mix(exp(x));
            mix(tanh(x));
            mix(sin(x));
            mix(cos(x));
            mix(ln(x.abs() + 1e-3));
            mix(powf(x.abs() + 1e-3, 0.37));
            mix(ln_1p(x.abs() * 1e-4));
        }
        assert_eq!(hash, 0xaf84_e60e_80f1_0180);
    }
}
