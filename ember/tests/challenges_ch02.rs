//! Tests for the Chapter 2 challenges: `cargo test --test challenges_ch02`.
use ember::Tensor;

fn floats(xs: &[usize]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}

#[test]
fn predict_four_views() {
    let x = Tensor::arange(&[4, 6]);
    let a = x.transpose(0, 1).slice(0, 1, 4, 2);
    assert_eq!(
        (a.shape(), a.strides(), a.offset()),
        (&[2, 4][..], &[2, 6][..], 1)
    );
    assert_eq!(a.to_vec(), floats(&[1, 7, 13, 19, 3, 9, 15, 21]));
    assert!(!a.is_contiguous());

    let b = x.select(1, 5);
    assert_eq!(
        (b.shape(), b.strides(), b.offset()),
        (&[4][..], &[6][..], 5)
    );
    assert_eq!(b.to_vec(), floats(&[5, 11, 17, 23]));

    let c = x.slice(0, 2, 4, 1).reshape(&[3, 4]);
    assert_eq!((c.strides(), c.offset()), (&[4, 1][..], 12));
    assert!(c.shares_storage(&x));

    let d = x.permute(&[1, 0]).contiguous();
    assert_eq!(d.strides(), &[4, 1]);
    assert!(!d.shares_storage(&x));
}

#[test]
fn unsqueeze_and_squeeze() {
    let x = Tensor::arange(&[2, 3]);
    let a = x.unsqueeze(0);
    assert_eq!(a.shape(), &[1, 2, 3]);
    assert!(a.is_contiguous() && a.shares_storage(&x));
    assert_eq!(x.unsqueeze(1).shape(), &[2, 1, 3]);
    assert_eq!(x.unsqueeze(2).shape(), &[2, 3, 1]);
    assert_eq!(x.unsqueeze(2).to_vec(), x.to_vec());
    assert_eq!(x.unsqueeze(1).squeeze(1).shape(), &[2, 3]);
    assert_eq!(
        x.transpose(0, 1).unsqueeze(0).to_vec(),
        x.transpose(0, 1).to_vec()
    );
}

#[test]
#[should_panic(expected = "squeeze: axis 0 has size 2")]
fn squeeze_rejects_a_real_axis() {
    Tensor::arange(&[2, 3]).squeeze(0);
}

#[test]
fn diagonal_is_a_view() {
    let x = Tensor::arange(&[3, 3]);
    let d = x.diagonal();
    assert_eq!((d.shape(), d.strides()), (&[3][..], &[4][..]));
    assert_eq!(d.to_vec(), floats(&[0, 4, 8]));
    assert!(d.shares_storage(&x));
    assert_eq!(
        Tensor::arange(&[3, 4]).diagonal().to_vec(),
        floats(&[0, 5, 10])
    );
    assert_eq!(Tensor::arange(&[4, 2]).diagonal().to_vec(), floats(&[0, 3]));
    assert_eq!(
        Tensor::arange(&[2, 3]).transpose(0, 1).diagonal().to_vec(),
        floats(&[0, 4])
    );
}

#[test]
fn chunk_splits_into_views() {
    let parts = Tensor::arange(&[10]).chunk(0, 3);
    let sizes: Vec<usize> = parts.iter().map(|p| p.shape()[0]).collect();
    assert_eq!(sizes, vec![4, 4, 2]);
    assert_eq!(parts[2].to_vec(), floats(&[8, 9]));

    let x = Tensor::arange(&[2, 6]);
    let cols = x.chunk(1, 3);
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[1].offset(), 2);
    assert_eq!(cols[1].to_vec(), floats(&[2, 3, 8, 9]));
    assert!(cols.iter().all(|c| c.shares_storage(&x)));
}

#[test]
fn unfold_makes_sliding_windows() {
    let x = Tensor::arange(&[6]);
    let w = x.unfold(0, 3, 1);
    assert_eq!((w.shape(), w.strides()), (&[4, 3][..], &[1, 1][..]));
    assert_eq!(w.to_vec(), floats(&[0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5]));
    let pairs = x.unfold(0, 2, 2);
    assert_eq!(pairs.shape(), &[3, 2]);
    assert!(pairs.is_contiguous());
    let m = Tensor::arange(&[2, 5]).unfold(1, 3, 2);
    assert_eq!((m.shape(), m.strides()), (&[2, 2, 3][..], &[5, 2, 1][..]));
    assert_eq!(m.to_vec(), floats(&[0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]));
}

#[test]
fn view_reshapes_without_copying_when_it_can() {
    let x = Tensor::arange(&[2, 3, 4]);
    let v = x
        .view(&[4, 6])
        .expect("a contiguous tensor always has a view");
    assert_eq!(v.strides(), &[6, 1]);

    let t = Tensor::arange(&[4, 6]).transpose(0, 1); // shape [6, 4], strides [1, 6]
    let v = t
        .view(&[2, 3, 4])
        .expect("splitting the first axis is a view");
    assert_eq!(v.strides(), &[3, 1, 6]);
    assert_eq!(v.to_vec(), t.reshape(&[2, 3, 4]).to_vec());
    assert!(v.shares_storage(&t));
    let v = t
        .view(&[6, 2, 2])
        .expect("splitting the second axis is a view");
    assert_eq!(v.strides(), &[1, 12, 6]);
    assert_eq!(v.to_vec(), t.reshape(&[6, 2, 2]).to_vec());
    assert!(t.view(&[24]).is_none());
    assert!(t.view(&[3, 8]).is_none());

    let s = Tensor::arange(&[4, 6]).slice(1, 0, 3, 1); // shape [4, 3], strides [6, 1]
    let v = s.view(&[2, 2, 3]).expect("splitting the rows is a view");
    assert_eq!(v.strides(), &[12, 6, 1]);
    assert_eq!(v.to_vec(), s.reshape(&[2, 2, 3]).to_vec());
    assert!(s.view(&[12]).is_none());
    assert_eq!(s.view(&[4, 1, 3]).map(|v| v.to_vec()), Some(s.to_vec()));
}
