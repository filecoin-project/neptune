// Allow `&Matrix` in function signatures.
#![allow(clippy::ptr_arg)]

use crate::field::PoseidonField;
use crate::matrix;
use crate::matrix::{
    apply_matrix, invert, is_identity, is_invertible, is_square, mat_mul, minor, transpose, Matrix,
};

#[derive(Clone, Debug, PartialEq)]
pub struct MdsMatrices<F: PoseidonField> {
    pub m: Matrix<F>,
    pub m_inv: Matrix<F>,
    pub m_hat: Matrix<F>,
    pub m_hat_inv: Matrix<F>,
    pub m_prime: Matrix<F>,
    pub m_double_prime: Matrix<F>,
}

pub fn create_mds_matrices<F: PoseidonField>(t: usize) -> MdsMatrices<F> {
    let m = generate_mds(t);
    derive_mds_matrices(m)
}

pub fn derive_mds_matrices<F: PoseidonField>(m: Matrix<F>) -> MdsMatrices<F> {
    let m_inv = invert(&m).unwrap(); // m is MDS so invertible.
    let m_hat = minor(&m, 0, 0);
    let m_hat_inv = invert(&m_hat).unwrap(); // If this returns None, then `mds_matrix` was not correctly generated.
    let m_prime = make_prime(&m);
    let m_double_prime = make_double_prime(&m, &m_hat_inv);

    MdsMatrices {
        m,
        m_inv,
        m_hat,
        m_hat_inv,
        m_prime,
        m_double_prime,
    }
}

/// A `SparseMatrix` is specifically one of the form of M''.
/// This means its first row and column are each dense, and the interior matrix
/// (minor to the element in both the row and column) is the identity.
/// We will pluralize this compact structure `sparse_matrixes` to distinguish from `sparse_matrices` from which they are created.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix<F: PoseidonField> {
    /// `w_hat` is the first column of the M'' matrix. It will be directly multiplied (scalar product) with a row of state elements.
    pub w_hat: Vec<F>,
    /// `v_rest` contains all but the first (already included in `w_hat`).
    pub v_rest: Vec<F>,
}

impl<F: PoseidonField> SparseMatrix<F> {
    pub fn new(m_double_prime: Matrix<F>) -> Self {
        assert!(Self::is_sparse_matrix(&m_double_prime));
        let size = matrix::rows(&m_double_prime);

        let w_hat = (0..size).map(|i| m_double_prime[i][0]).collect::<Vec<_>>();
        let v_rest = m_double_prime[0][1..].to_vec();

        Self { w_hat, v_rest }
    }

    pub fn is_sparse_matrix(m: &Matrix<F>) -> bool {
        is_square(&m) && is_identity(&minor(&m, 0, 0))
    }

    pub fn size(&self) -> usize {
        self.w_hat.len()
    }

    pub fn to_matrix(&self) -> Matrix<F> {
        let mut m = matrix::make_identity(self.size());
        for (j, elt) in self.w_hat.iter().enumerate() {
            m[j][0] = *elt;
        }
        for (i, elt) in self.v_rest.iter().enumerate() {
            m[0][i + 1] = *elt;
        }
        m
    }
}

// - Having effectively moved the round-key additions into the S-boxes, refactor MDS matrices used for partial-round mix layer to use sparse matrices.
// - This requires using a different (sparse) matrix at each partial round, rather than the same dense matrix at each.
//   - The MDS matrix, M, for each such round, starting from the last, is factored into two components, such that M' x M'' = M.
//   - M'' is sparse and replaces M for the round.
//   - The previous layer's M is then replaced by M x M' = M*.
//   - M* is likewise factored into M*' and M*'', and the process continues.
pub fn factor_to_sparse_matrixes<F: PoseidonField>(
    base_matrix: Matrix<F>,
    n: usize,
) -> (Matrix<F>, Vec<SparseMatrix<F>>) {
    let (pre_sparse, sparse_matrices) = factor_to_sparse_matrices(base_matrix, n);
    let sparse_matrixes = sparse_matrices
        .iter()
        .map(|m| SparseMatrix::<F>::new(m.to_vec()))
        .collect::<Vec<_>>();

    (pre_sparse, sparse_matrixes)
}

pub fn factor_to_sparse_matrices<F: PoseidonField>(
    base_matrix: Matrix<F>,
    n: usize,
) -> (Matrix<F>, Vec<Matrix<F>>) {
    let (pre_sparse, mut all) =
        (0..n).fold((base_matrix.clone(), Vec::new()), |(curr, mut acc), _| {
            let derived = derive_mds_matrices(curr);
            acc.push(derived.m_double_prime);
            let new = mat_mul(&base_matrix, &derived.m_prime).unwrap();
            (new, acc)
        });
    all.reverse();
    (pre_sparse, all)
}

fn generate_mds<F: PoseidonField>(t: usize) -> Matrix<F> {
    // Source: https://github.com/dusk-network/dusk-poseidon-merkle/commit/776c37734ea2e71bb608ce4bc58fdb5f208112a7#diff-2eee9b20fb23edcc0bf84b14167cbfdc
    // Generate x and y values deterministically for the cauchy matrix
    // where x[i] != y[i] to allow the values to be inverted
    // and there are no duplicates in the x vector or y vector, so that the determinant is always non-zero
    // [a b]
    // [c d]
    // det(M) = (ad - bc) ; if a == b and c == d => det(M) =0
    // For an MDS matrix, every possible mxm submatrix, must have det(M) != 0
    let xs: Vec<F> = (0..t as u64).map(F::from_u64).collect();
    let ys: Vec<F> = (t as u64..2 * t as u64).map(F::from_u64).collect();

    let matrix = xs
        .iter()
        .map(|xs_item| {
            ys.iter()
                .map(|ys_item| {
                    // Generate the entry at (i,j)
                    let mut tmp = *xs_item;
                    tmp.add_assign(&ys_item);
                    tmp.inverse().unwrap()
                })
                .collect()
        })
        .collect();

    // To ensure correctness, we would check all sub-matrices for invertibility. Meanwhile, this is a simple sanity check.
    assert!(is_invertible(&matrix));

    //  `poseidon::product_mds_with_matrix` relies on the constructed MDS matrix being symmetric, so ensure it is.
    assert_eq!(matrix, transpose(&matrix));
    matrix
}

fn make_prime<F: PoseidonField>(m: &Matrix<F>) -> Matrix<F> {
    m.iter()
        .enumerate()
        .map(|(i, row)| match i {
            0 => {
                let mut new_row = vec![F::zero(); row.len()];
                new_row[0] = F::one();
                new_row
            }
            _ => {
                let mut new_row = vec![F::zero(); row.len()];
                new_row[1..].copy_from_slice(&row[1..]);
                new_row
            }
        })
        .collect()
}

fn make_double_prime<F: PoseidonField>(m: &Matrix<F>, m_hat_inv: &Matrix<F>) -> Matrix<F> {
    let (v, w) = make_v_w(m);
    let w_hat = apply_matrix(m_hat_inv, &w);

    m.iter()
        .enumerate()
        .map(|(i, row)| match i {
            0 => {
                let mut new_row = Vec::with_capacity(row.len());
                new_row.push(row[0]);
                new_row.extend(&v);
                new_row
            }
            _ => {
                let mut new_row = vec![F::zero(); row.len()];
                new_row[0] = w_hat[i - 1];
                new_row[i] = F::one();
                new_row
            }
        })
        .collect()
}

fn make_v_w<F: PoseidonField>(m: &Matrix<F>) -> (Vec<F>, Vec<F>) {
    let v = m[0][1..].to_vec();
    let w = m.iter().skip(1).map(|column| column[0]).collect();
    (v, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use bellperson::bls::Fr;
    use fff::Field;
    use matrix::left_apply_matrix;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_mds_matrices_creation() {
        for i in 2..5 {
            test_mds_matrices_creation_aux(i);
        }
    }

    fn test_mds_matrices_creation_aux(width: usize) {
        let MdsMatrices {
            m,
            m_inv,
            m_hat,
            m_hat_inv: _,
            m_prime,
            m_double_prime,
        } = create_mds_matrices::<Fr>(width);

        for i in 0..m_hat.len() {
            for j in 0..m_hat[i].len() {
                assert_eq!(m[i + 1][j + 1], m_hat[i][j], "MDS minor has wrong value.");
            }
        }

        // M^-1 x M = I
        assert!(matrix::is_identity(&matrix::mat_mul(&m_inv, &m).unwrap()));

        // M' x M'' = M
        assert_eq!(m, matrix::mat_mul(&m_prime, &m_double_prime).unwrap());
    }

    #[test]
    fn test_swapping() {
        test_swapping_aux(3);
    }

    fn test_swapping_aux(width: usize) {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let MdsMatrices {
            m,
            m_inv: _,
            m_hat: _,
            m_hat_inv: _,
            m_prime,
            m_double_prime,
        } = create_mds_matrices::<Fr>(width);

        let mut base = Vec::with_capacity(width);
        for _ in 0..width {
            base.push(Fr::random(&mut rng));
        }

        let mut x = base.clone();
        x[0] = Fr::random(&mut rng);

        let mut y = base;
        y[0] = Fr::random(&mut rng);

        let qx = apply_matrix(&m_prime, &x);
        let qy = apply_matrix(&m_prime, &y);
        assert_eq!(qx[0], x[0]);
        assert_eq!(qy[0], y[0]);
        assert_eq!(qx[1..], qy[1..]);

        let mx = left_apply_matrix(&m, &x);
        let m1_m2_x = left_apply_matrix(&m_prime, &left_apply_matrix(&m_double_prime, &x));
        assert_eq!(mx, m1_m2_x);

        let xm = apply_matrix(&m, &x);
        let x_m1_m2 = apply_matrix(&m_double_prime, &apply_matrix(&m_prime, &x));
        assert_eq!(xm, x_m1_m2);

        let mut rk = Vec::with_capacity(width);
        for _ in 0..width {
            rk.push(Fr::random(&mut rng));
        }
    }

    #[test]
    fn test_factor_to_sparse_matrices() {
        for width in 3..9 {
            test_factor_to_sparse_matrices_aux(width, 3);
        }
    }

    fn test_factor_to_sparse_matrices_aux(width: usize, n: usize) {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let m = generate_mds::<Fr>(width);
        let m2 = m.clone();

        let (pre_sparse, mut sparse) = factor_to_sparse_matrices(m, n);
        assert_eq!(n, sparse.len());

        let mut initial = Vec::with_capacity(width);
        for _ in 0..width {
            initial.push(Fr::random(&mut rng));
        }

        let mut round_keys = Vec::with_capacity(width);
        for _ in 0..(n + 1) {
            round_keys.push(Fr::random(&mut rng));
        }

        let expected = std::iter::repeat(m2).take(n + 1).zip(&round_keys).fold(
            initial.clone(),
            |mut acc, (m, rk)| {
                acc = apply_matrix(&m, &acc);
                quintic_s_box(&mut acc[0], None, Some(&rk));
                acc
            },
        );

        sparse.insert(0, pre_sparse);
        let actual = sparse
            .iter()
            .zip(&round_keys)
            .fold(initial, |mut acc, (m, rk)| {
                acc = apply_matrix(&m, &acc);
                quintic_s_box(&mut acc[0], None, Some(&rk));
                acc
            });
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_factor_to_sparse_matrixes() {
        for width in 3..9 {
            test_factor_to_sparse_matrixes_aux(width, 3);
        }
    }

    fn test_factor_to_sparse_matrixes_aux(width: usize, n: usize) {
        let m = generate_mds::<Fr>(width);
        let m2 = m.clone();

        let (pre_sparse, sparse_matrices) = factor_to_sparse_matrices(m, n);
        assert_eq!(n, sparse_matrices.len());

        let (pre_sparse2, sparse_matrixes) = factor_to_sparse_matrixes(m2, n);

        assert_eq!(pre_sparse, pre_sparse2);

        let matrices_again = sparse_matrixes
            .iter()
            .map(|m| m.to_matrix())
            .collect::<Vec<_>>();
        dbg!(&sparse_matrixes, &sparse_matrices);

        let _ = sparse_matrices
            .iter()
            .zip(matrices_again.iter())
            .map(|(a, b)| {
                dbg!(&a, &b);
                assert_eq!(a, b)
            })
            .collect::<Vec<_>>();

        assert_eq!(sparse_matrices, matrices_again);
    }
}
