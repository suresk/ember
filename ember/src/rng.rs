//! Chapter 12: a seeded random number generator, so that every random tensor in ember
//! (initial weights, shuffled batches, dropout masks) can be made again, bit for bit.

use crate::math;
use crate::tensor::Tensor;

// @listing rng
/// SplitMix64 (Steele, Lea and Flood, 2014): a counter that steps by a fixed odd
/// number, and a mixing function that scrambles each count into 64 random bits.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15); // 2⁶⁴ / φ
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
// @end

// @listing uniform
impl Rng {
    /// Uniform in [0, 1): the top 24 bits, as many as an `f32` holds, times 2⁻²⁴.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform between `lo` and `hi`.
    pub fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform in [0, 1) with 53 bits, as many as an `f64` holds.
    fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
// @end

// @listing normal
impl Rng {
    /// A sample of the standard normal distribution, by the Box–Muller transform.
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.uniform_f64(); // in (0, 1], so ln(u1) is finite
        let u2 = self.uniform_f64();
        let radius = (-2.0 * math::ln64(u1)).sqrt();
        (radius * math::cos64(std::f64::consts::TAU * u2)) as f32
    }
}

impl Tensor {
    /// Uniform samples in [0, 1), in row-major order.
    pub fn rand(shape: &[usize], rng: &mut Rng) -> Tensor {
        let n = shape.iter().product();
        Tensor::from_vec((0..n).map(|_| rng.uniform()).collect(), shape)
    }

    /// Standard normal samples, in row-major order.
    pub fn randn(shape: &[usize], rng: &mut Rng) -> Tensor {
        let n = shape.iter().product();
        Tensor::from_vec((0..n).map(|_| rng.normal()).collect(), shape)
    }
}
// @end

// @listing shuffle
impl Rng {
    /// A uniform index in `0..n`: the top 64 bits of `next_u64() · n`.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Fisher–Yates: each position, from the last, swaps with a random one at or
    /// before it, so each of the `n!` orders comes out with probability `1/n!`.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
}
// @end

// @listing state
impl Rng {
    /// The generator's whole state: `Rng::new(rng.state())` continues the stream
    /// exactly where `rng` is (Chapter 20's checkpoints save it).
    pub fn state(&self) -> u64 {
        self.state
    }
}
// @end

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_rng_is_eight_bytes() {
        assert_eq!(size_of::<Rng>(), 8);
    }
}
