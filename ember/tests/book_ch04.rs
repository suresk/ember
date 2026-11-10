//! The examples printed in Chapter 4 of the guided edition.
use ember::Tensor;

/// Integers as `f32`s, to compare with `to_vec`.
fn floats(xs: &[i32]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

#[test]
fn map_examples() {
    // @listing map-example
    let x = Tensor::arange(&[2, 3]);
    let scale = 0.5;
    let y = x.map(|v| v * scale); // the closure captures `scale`
    assert_eq!(y.to_vec(), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    assert!(!y.shares_storage(&x)); // map always builds a new buffer

    let t = x.transpose(0, 1); // not contiguous: the general path
    let z = t.map(|v| 10.0 * v);
    assert_eq!(z.to_vec(), floats(&[0, 30, 10, 40, 20, 50]));
    assert_eq!(z.strides(), &[2, 1]); // the result is a fresh row-major tensor
    // @end
}

#[test]
fn unary_examples() {
    let t = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 1.5], &[4]);
    assert_eq!(t.neg().to_vec(), vec![2.0, 0.5, -0.0, -1.5]);
    assert_eq!(
        t.mul_scalar(2.0).add_scalar(1.0).to_vec(),
        vec![-3.0, 0.0, 1.0, 4.0]
    );
    // @listing unary-example
    assert_eq!(t.relu().to_vec(), vec![0.0, 0.0, 0.0, 1.5]);
    assert_eq!(t.powf(2.0).to_vec(), vec![4.0, 0.25, 0.0, 2.25]);
    let s = Tensor::from_vec(vec![-100.0, 0.0, 100.0], &[3]);
    assert_eq!(s.sigmoid().to_vec(), vec![0.0, 0.5, 1.0]); // e^100 is inf in f32
    // @end
    assert_eq!(s.tanh().to_vec(), vec![-1.0, 0.0, 1.0]);
    let e = Tensor::from_vec(vec![1.0, 4.0], &[2]);
    assert_eq!(e.ln().exp().to_vec(), vec![1.0, 4.0]);
    assert_eq!(e.sqrt().to_vec(), vec![1.0, 2.0]);
}

#[test]
fn which_tensors_take_the_fast_path() {
    // @listing as-slice-example
    let x = Tensor::arange(&[2, 3]);
    assert_eq!(x.as_slice(), Some(&floats(&[0, 1, 2, 3, 4, 5])[..]));
    let row = x.select(0, 1); // offset 3, stride 1: still one run
    assert_eq!(row.as_slice(), Some(&floats(&[3, 4, 5])[..]));
    assert_eq!(x.transpose(0, 1).as_slice(), None);

    let b = Tensor::arange(&[3]);
    assert_eq!(b.expand(&[2, 3]).as_slice(), None); // stride 0 on a real axis
    assert!(b.expand(&[1, 3]).as_slice().is_some()); // stride 0 on a size-1 axis
    // @end
}

#[test]
fn adding_with_broadcasting() {
    // @listing add-example
    let x = Tensor::arange(&[2, 3]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
    assert_eq!(x.add(&b).to_vec(), floats(&[10, 21, 32, 13, 24, 35]));
    assert_eq!(x.add(&b).shape(), &[2, 3]);

    // The fast path and the general path agree.
    let t = x.transpose(0, 1).contiguous().transpose(0, 1); // x's values, strided
    assert_eq!(t.as_slice(), None);
    assert_eq!(t.mul(&x).to_vec(), x.mul(&x).to_vec());
    // @end
    assert_eq!(x.div(&b).get(&[1, 2]), 5.0 / 30.0);
}

// @listing add-panics
#[test]
#[should_panic(expected = "add: shapes [2, 3] and [2] don't broadcast")]
fn add_names_the_operation_and_both_shapes() {
    Tensor::arange(&[2, 3]).add(&Tensor::arange(&[2]));
}
// @end

#[test]
fn zip_map_example() {
    // @listing zip-map-example
    let x = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3]);
    let y = Tensor::from_vec(vec![4.0, 2.0, 3.0], &[3]);
    let bigger = x.zip_map(&y, f32::max);
    assert_eq!(bigger.to_vec(), vec![4.0, 5.0, 3.0]);
    let hypot = x.zip_map(&y, |a, b| (a * a + b * b).sqrt());
    assert_eq!(hypot.get(&[0]), 17f32.sqrt());
    // @end
}

#[test]
fn operators_on_references() {
    // @listing operators-example
    let x = Tensor::arange(&[2, 3]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
    assert_eq!((&x + &b).to_vec(), floats(&[10, 21, 32, 13, 24, 35]));
    assert_eq!((&x * 2.0).to_vec(), floats(&[0, 2, 4, 6, 8, 10]));
    assert_eq!((-&x).get(&[1, 2]), -5.0);

    // x² - x: the inner result is a temporary, borrowed with `&( … )`.
    let y = &(&x * &x) - &x;
    assert_eq!(y.to_vec(), floats(&[0, 0, 2, 6, 12, 20]));
    assert_eq!(x.get(&[1, 2]), 5.0); // x and b are still ours to use
    // @end
}

#[test]
fn reductions_on_a_matrix() {
    // @listing reduce-example
    let x = Tensor::arange(&[2, 3]); // [[0, 1, 2], [3, 4, 5]]
    let s0 = x.sum(0, false);
    assert_eq!((s0.shape(), s0.to_vec()), (&[3][..], floats(&[3, 5, 7])));
    let s1 = x.sum(1, false);
    assert_eq!((s1.shape(), s1.to_vec()), (&[2][..], floats(&[3, 12])));

    assert_eq!(x.sum(1, true).shape(), &[2, 1]); // keepdim: the axis stays
    // @end
    assert_eq!(x.sum(0, true).shape(), &[1, 3]);
    assert_eq!(x.mean(0, false).to_vec(), vec![1.5, 2.5, 3.5]);
    assert_eq!(x.mean(1, false).to_vec(), vec![1.0, 4.0]);
    assert_eq!(x.max(1, false).to_vec(), vec![2.0, 5.0]);
    assert_eq!(x.argmax(0), vec![1, 1, 1]);
}

#[test]
fn reductions_on_a_view() {
    // @listing reduce-3d
    let x = Tensor::arange(&[2, 3, 4]);
    let s = x.sum(1, false); // shape [2, 4]
    assert_eq!(s.shape(), &[2, 4]);
    assert_eq!(s.to_vec(), floats(&[12, 15, 18, 21, 48, 51, 54, 57]));

    // A transpose reduces like the copy it stands for.
    let t = Tensor::arange(&[2, 3]).transpose(0, 1); // [[0, 3], [1, 4], [2, 5]]
    assert_eq!(t.sum(1, false).to_vec(), floats(&[3, 5, 7]));
    // @end

    // Empty lanes: a sum of nothing is 0, a mean of nothing is 0 / 0.
    let empty = Tensor::zeros(&[2, 0]);
    assert_eq!(empty.sum(1, false).to_vec(), vec![0.0, 0.0]);
    assert!(empty.mean(1, false).get(&[0]).is_nan());
    assert_eq!(empty.sum(0, false).shape(), &[0]);
}

#[test]
fn keepdim_lets_the_result_broadcast_back() {
    // @listing center-rows
    let x = Tensor::arange(&[2, 3]);
    let m = x.mean(1, true); // [[1], [4]], shape [2, 1]
    assert_eq!((&x - &m).to_vec(), floats(&[-1, 0, 1, -1, 0, 1]));

    let c = x.mean(0, false); // [1.5, 2.5, 3.5], shape [3]: no keepdim needed
    assert_eq!((&x - &c).to_vec(), vec![-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]);
    // @end
}

// @listing center-panics
#[test]
#[should_panic(expected = "sub: shapes [2, 3] and [2] don't broadcast")]
fn row_means_without_keepdim_do_not_broadcast() {
    let x = Tensor::arange(&[2, 3]);
    let _ = &x - &x.mean(1, false);
}
// @end

#[test]
fn sums_are_accumulated_in_f64() {
    // @listing f64-sum
    let xs = vec![16_777_216.0, 1.0, 1.0, 1.0, 1.0]; // 2^24 and four 1s
    let naive: f32 = xs.iter().sum(); // each 2^24 + 1 rounds back to 2^24
    assert_eq!(naive, 16_777_216.0);
    let t = Tensor::from_vec(xs, &[5]);
    assert_eq!(t.sum_all(), 16_777_220.0);

    let big = Tensor::from_vec(vec![3e38, 3e38], &[2]);
    assert_eq!(big.sum_all(), f32::INFINITY); // 6e38 is too big for f32
    assert_eq!(big.mean(0, false).item(), 3e38); // the mean isn't
    // @end
}

#[test]
fn max_and_argmax() {
    // @listing max-example
    let t = Tensor::from_vec(vec![-3.0, -1.0, -2.0], &[3]);
    assert_eq!(t.max(0, false).item(), -1.0);
    let wrong = t.to_vec().into_iter().fold(0.0, f32::max); // starting from 0
    assert_eq!(wrong, 0.0);

    let ties = Tensor::from_vec(vec![1.0, 3.0, 3.0, 7.0, 7.0, 7.0], &[2, 3]);
    assert_eq!(ties.argmax(1), vec![1, 0]); // the first of the equal values
    // @end

    let nan = Tensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3]);
    assert!(nan.max(0, false).item().is_nan());
    assert_eq!(f32::NAN.max(3.0), 3.0); // f32::max hides the NaN
}

#[test]
fn sum_all_and_item() {
    // @listing item-example
    let x = Tensor::arange(&[2, 3]);
    assert_eq!(x.transpose(0, 1).sum_all(), 15.0);
    let s = x.sum(0, false).sum(0, false); // shape []
    assert_eq!((s.ndim(), s.item()), (0, 15.0));
    assert_eq!(x.sum(1, true).sum(0, true).item(), 15.0); // shape [1, 1]
    // @end
    assert_eq!(x.sum_all(), 15.0);
    assert_eq!(x.select(0, 1).select(0, 2).item(), 5.0); // a view, offset 5
}

// @listing item-panics
#[test]
#[should_panic(expected = "item: a tensor of shape [2, 3] has 6 elements, not 1")]
fn item_needs_exactly_one_element() {
    Tensor::arange(&[2, 3]).item();
}
// @end

#[test]
fn sum_to_undoes_expand() {
    // @listing sum-to-example
    let y = Tensor::arange(&[2, 3]);
    let a = y.sum_to(&[3]); // the axis added on the left is summed away
    assert_eq!((a.shape(), a.to_vec()), (&[3][..], floats(&[3, 5, 7])));
    let b = y.sum_to(&[2, 1]); // a stretched axis is summed, and kept
    assert_eq!((b.shape(), b.to_vec()), (&[2, 1][..], floats(&[3, 12])));
    assert_eq!(y.sum_to(&[1, 3]).shape(), &[1, 3]);
    assert_eq!(y.sum_to(&[]).item(), 15.0);
    assert_eq!(y.sum_to(&[2, 3]).to_vec(), y.to_vec()); // nothing to undo
    // @end

    // @listing sum-to-roundtrip
    let bias = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
    let e = bias.expand(&[2, 3]);
    assert_eq!(e.sum_to(&[3]).to_vec(), floats(&[20, 40, 60])); // two reads each

    let grid =
        Tensor::from_vec(vec![0.0, 10.0, 20.0], &[3, 1]).add(&Tensor::arange(&[4]));
    assert_eq!(grid.sum_to(&[3, 1]).to_vec(), floats(&[6, 46, 86]));
    assert_eq!(grid.sum_to(&[4]).to_vec(), floats(&[30, 33, 36, 39]));
    // @end
}

// @listing sum-to-panics
#[test]
#[should_panic(expected = "sum_to: [2] does not expand to [2, 3]")]
fn sum_to_rejects_a_shape_that_does_not_expand() {
    Tensor::arange(&[2, 3]).sum_to(&[2]);
}
// @end
