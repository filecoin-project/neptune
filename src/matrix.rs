// Allow `&Matrix` in function signatures.
#![allow(clippy::ptr_arg)]

use crate::field::PoseidonField;

/// Matrix functions here are, at least for now, quick and dirty — intended only to support precomputation of poseidon optimization.

/// Matrix represented as a Vec of rows, so that m[i][j] represents the jth column of the ith row in Matrix, m.
pub type Matrix<T> = Vec<Vec<T>>;

pub fn rows<T>(matrix: &Matrix<T>) -> usize {
    matrix.len()
}

/// Panics if `matrix` is not actually a matrix. So only use any of these functions on well-formed data.
/// Only use during constant calculation on matrices known to have been constructed correctly.
fn columns<T>(matrix: &Matrix<T>) -> usize {
    if matrix.is_empty() {
        0
    } else {
        let column_length = matrix[0].len();
        for row in matrix {
            if row.len() != column_length {
                panic!("not a matrix");
            }
        }
        column_length
    }
}

// This wastefully discards the actual inverse, if it exists, so in general callers should
// just call `invert` if that result will be needed.
pub(crate) fn is_invertible<F: PoseidonField>(matrix: &Matrix<F>) -> bool {
    is_square(matrix) && invert(matrix).is_some()
}

fn scalar_mul<F: PoseidonField>(scalar: F, matrix: &Matrix<F>) -> Matrix<F> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|val| {
                    let mut prod = scalar;
                    prod.mul_assign(val);
                    prod
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn scalar_vec_mul<F: PoseidonField>(scalar: F, vec: &[F]) -> Vec<F> {
    vec.iter()
        .map(|val| {
            let mut prod = scalar;
            prod.mul_assign(val);
            prod
        })
        .collect::<Vec<_>>()
}

pub fn mat_mul<F: PoseidonField>(a: &Matrix<F>, b: &Matrix<F>) -> Option<Matrix<F>> {
    if rows(a) != columns(b) {
        return None;
    };

    let b_t = transpose(b);

    let res = a
        .iter()
        .map(|input_row| {
            b_t.iter()
                .map(|transposed_column| vec_mul(&input_row, &transposed_column))
                .collect()
        })
        .collect();

    Some(res)
}

fn vec_mul<F: PoseidonField>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |mut acc, (v1, v2)| {
        let mut tmp = *v1;
        tmp.mul_assign(&v2);
        acc.add_assign(&tmp);
        acc
    })
}

pub fn vec_add<F: PoseidonField>(a: &[F], b: &[F]) -> Vec<F> {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let mut res = *a;
            res.add_assign(b);
            res
        })
        .collect::<Vec<_>>()
}

pub fn vec_sub<F: PoseidonField>(a: &[F], b: &[F]) -> Vec<F> {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let mut res = *a;
            res.sub_assign(b);
            res
        })
        .collect::<Vec<_>>()
}

/// Left-multiply a vector by a square matrix of same size: MV where V is considered a column vector.
pub fn left_apply_matrix<F: PoseidonField>(m: &Matrix<F>, v: &[F]) -> Vec<F> {
    assert!(is_square(m), "Only square matrix can be applied to vector.");
    assert_eq!(
        rows(m),
        v.len(),
        "Matrix can only be applied to vector of same size."
    );

    let mut result: Vec<F> = vec![F::zero(); v.len()];

    for (result, row) in result.iter_mut().zip(m.iter()) {
        for (mat_val, vec_val) in row.iter().zip(v) {
            let mut tmp = *mat_val;
            tmp.mul_assign(vec_val);
            result.add_assign(&tmp);
        }
    }
    result
}

/// Right-multiply a vector by a square matrix  of same size: VM where V is considered a row vector.
pub fn apply_matrix<F: PoseidonField>(m: &Matrix<F>, v: &[F]) -> Vec<F> {
    assert!(is_square(m), "Only square matrix can be applied to vector.");
    assert_eq!(
        rows(m),
        v.len(),
        "Matrix can only be applied to vector of same size."
    );

    let mut result: Vec<F> = vec![F::zero(); v.len()];
    for (j, val) in result.iter_mut().enumerate() {
        for (i, row) in m.iter().enumerate() {
            let mut tmp = row[j];
            tmp.mul_assign(&v[i]);
            val.add_assign(&tmp);
        }
    }

    result
}

#[allow(clippy::needless_range_loop)]
pub fn transpose<F: PoseidonField>(matrix: &Matrix<F>) -> Matrix<F> {
    let size = rows(matrix);
    let mut new = Vec::with_capacity(size);
    for j in 0..size {
        let mut row = Vec::with_capacity(size);
        for i in 0..size {
            row.push(matrix[i][j])
        }
        new.push(row);
    }
    new
}

#[allow(clippy::needless_range_loop)]
pub fn make_identity<F: PoseidonField>(size: usize) -> Matrix<F> {
    let mut result = vec![vec![F::zero(); size]; size];
    for i in 0..size {
        result[i][i] = F::one();
    }
    result
}

pub fn kronecker_delta<F: PoseidonField>(i: usize, j: usize) -> F {
    if i == j {
        F::one()
    } else {
        F::zero()
    }
}

pub fn is_identity<F: PoseidonField>(matrix: &Matrix<F>) -> bool {
    for i in 0..rows(matrix) {
        for j in 0..columns(matrix) {
            if matrix[i][j] != kronecker_delta(i, j) {
                return false;
            }
        }
    }
    true
}

pub fn is_square<T>(matrix: &Matrix<T>) -> bool {
    rows(matrix) == columns(matrix)
}

pub fn minor<F: PoseidonField>(matrix: &Matrix<F>, i: usize, j: usize) -> Matrix<F> {
    assert!(is_square(matrix));
    let size = rows(matrix);
    assert!(size > 0);
    let new = matrix
        .iter()
        .enumerate()
        .filter_map(|(ii, row)| {
            if ii == i {
                None
            } else {
                let mut new_row = row.clone();
                new_row.remove(j);
                Some(new_row)
            }
        })
        .collect();
    assert!(is_square(&new));
    new
}

// Assumes matrix is partially reduced to upper triangular. `column` is the column to eliminate from all rows.
// Returns `None` if either:
//   - no non-zero pivot can be found for `column`
//   - `column` is not the first
fn eliminate<F: PoseidonField>(
    matrix: &Matrix<F>,
    column: usize,
    shadow: &mut Matrix<F>,
) -> Option<Matrix<F>> {
    let zero = F::zero();
    let pivot_index = (0..rows(matrix))
        .find(|&i| matrix[i][column] != zero && (0..column).all(|j| matrix[i][j] == zero))?;

    let pivot = &matrix[pivot_index];
    let pivot_val = pivot[column];

    let inv_pivot = pivot_val.inverse()?; // This should never fail since we have a non-zero `pivot_val` if we got here.
    let mut result = Vec::with_capacity(matrix.len());
    result.push(pivot.clone());

    for (i, row) in matrix.iter().enumerate() {
        if i == pivot_index {
            continue;
        };
        let val = row[column];
        if val == zero {
            // Value is already eliminated.
            result.push(row.to_vec());
        } else {
            let mut factor = val;
            factor.mul_assign(&inv_pivot);

            let scaled_pivot = scalar_vec_mul(factor, &pivot);
            let eliminated = vec_sub(row, &scaled_pivot);
            result.push(eliminated);

            let shadow_pivot = &shadow[pivot_index];
            let scaled_shadow_pivot = scalar_vec_mul(factor, shadow_pivot);
            let shadow_row = &shadow[i];
            shadow[i] = vec_sub(shadow_row, &scaled_shadow_pivot);
        }
    }
    Some(result)
}

// `matrix` must be square.
fn upper_triangular<F: PoseidonField>(
    matrix: &Matrix<F>,
    mut shadow: &mut Matrix<F>,
) -> Option<Matrix<F>> {
    assert!(is_square(matrix));
    let mut result = Vec::with_capacity(matrix.len());
    let mut shadow_result = Vec::with_capacity(matrix.len());

    let mut curr = matrix.clone();
    let mut column = 0;
    while curr.len() > 1 {
        let initial_rows = curr.len();

        curr = eliminate(&curr, column, &mut shadow)?;
        result.push(curr[0].clone());
        shadow_result.push(shadow[0].clone());
        column += 1;

        curr = curr[1..].to_vec();
        *shadow = shadow[1..].to_vec();
        assert_eq!(curr.len(), initial_rows - 1);
    }
    result.push(curr[0].clone());
    shadow_result.push(shadow[0].clone());

    *shadow = shadow_result;

    Some(result)
}

// `matrix` must be upper triangular.
fn reduce_to_identity<F: PoseidonField>(
    matrix: &Matrix<F>,
    shadow: &mut Matrix<F>,
) -> Option<Matrix<F>> {
    let size = rows(matrix);
    let mut result: Matrix<F> = Vec::new();
    let mut shadow_result: Matrix<F> = Vec::new();

    for i in 0..size {
        let idx = size - i - 1;
        let row = &matrix[idx];
        let shadow_row = &shadow[idx];

        let val = row[idx];
        let inv = val.inverse()?; // If `val` is zero, then there is no inverse, and we cannot compute a result.

        let mut normalized = scalar_vec_mul(inv, &row);
        let mut shadow_normalized = scalar_vec_mul(inv, &shadow_row);

        for j in 0..i {
            let idx = size - j - 1;
            let val = normalized[idx];
            let subtracted = scalar_vec_mul(val, &result[j]);
            let result_subtracted = scalar_vec_mul(val, &shadow_result[j]);

            normalized = vec_sub(&normalized, &subtracted);
            shadow_normalized = vec_sub(&shadow_normalized, &result_subtracted);
        }

        result.push(normalized);
        shadow_result.push(shadow_normalized);
    }

    result.reverse();
    shadow_result.reverse();

    *shadow = shadow_result;
    Some(result)
}

//
pub(crate) fn invert<F: PoseidonField>(matrix: &Matrix<F>) -> Option<Matrix<F>> {
    let mut shadow = make_identity(columns(matrix));
    let ut = upper_triangular(&matrix, &mut shadow);

    ut.and_then(|x| reduce_to_identity(&x, &mut shadow))
        .and(Some(shadow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellperson::bls::Fr;

    #[test]
    fn test_minor() {
        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);
        let seven = Fr::from_u64(7);
        let eight = Fr::from_u64(8);
        let nine = Fr::from_u64(9);

        let m = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, nine],
        ];

        let cases = [
            (0, 0, vec![vec![five, six], vec![eight, nine]]),
            (0, 1, vec![vec![four, six], vec![seven, nine]]),
            (0, 2, vec![vec![four, five], vec![seven, eight]]),
            (1, 0, vec![vec![two, three], vec![eight, nine]]),
            (1, 1, vec![vec![one, three], vec![seven, nine]]),
            (1, 2, vec![vec![one, two], vec![seven, eight]]),
            (2, 0, vec![vec![two, three], vec![five, six]]),
            (2, 1, vec![vec![one, three], vec![four, six]]),
            (2, 2, vec![vec![one, two], vec![four, five]]),
        ];
        for (i, j, expected) in &cases {
            let result = minor(&m, *i, *j);

            assert_eq!(*expected, result);
        }
    }

    #[test]
    fn test_scalar_mul() {
        let zero = Fr::from_u64(0);
        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let six = Fr::from_u64(6);

        let m = vec![vec![zero, one], vec![two, three]];
        let res = scalar_mul(two, &m);

        let expected = vec![vec![zero, two], vec![four, six]];

        assert_eq!(expected, res);
    }

    #[test]
    fn test_vec_mul() {
        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);

        let a = vec![one, two, three];
        let b = vec![four, five, six];
        let res = vec_mul(&a, &b);

        let expected = Fr::from_u64(32);

        assert_eq!(expected, res);
    }

    #[test]
    fn test_transpose() {
        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);
        let seven = Fr::from_u64(7);
        let eight = Fr::from_u64(8);
        let nine = Fr::from_u64(9);

        let m = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, nine],
        ];

        let expected = vec![
            vec![one, four, seven],
            vec![two, five, eight],
            vec![three, six, nine],
        ];

        let res = transpose(&m);
        assert_eq!(expected, res);
    }

    #[test]
    fn test_inverse() {
        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);
        let seven = Fr::from_u64(7);
        let eight = Fr::from_u64(8);
        let nine = Fr::from_u64(9);

        let m = vec![
            vec![one, two, three],
            vec![four, three, six],
            vec![five, eight, seven],
        ];

        let m1 = vec![
            vec![one, two, three],
            vec![four, five, six],
            vec![seven, eight, nine],
        ];

        assert!(!is_invertible(&m1));
        assert!(is_invertible(&m));

        let m_inv = invert(&m).unwrap();

        let computed_identity = mat_mul(&m, &m_inv).unwrap();
        assert!(is_identity(&computed_identity));

        // S
        let some_vec = vec![six, five, four];

        // M^-1(S)
        let inverse_applied = super::apply_matrix(&m_inv, &some_vec);

        // M(M^-1(S))
        let m_applied_after_inverse = super::apply_matrix(&m, &inverse_applied);

        // S = M(M^-1(S))
        assert_eq!(
            some_vec, m_applied_after_inverse,
            "M(M^-1(V))) = V did not hold"
        );

        //panic!();
        // B
        let base_vec = vec![eight, two, five];

        // S + M(B)
        let add_after_apply = vec_add(&some_vec, &apply_matrix(&m, &base_vec));

        // M(B + M^-1(S))
        let apply_after_add = apply_matrix(&m, &vec_add(&base_vec, &inverse_applied));

        // S + M(B) = M(B + M^-1(S))
        assert_eq!(add_after_apply, apply_after_add, "breakin' the law");
    }

    #[test]
    fn test_eliminate() {
        //let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);
        let seven = Fr::from_u64(7);
        let eight = Fr::from_u64(8);
        //        let nine = Fr::from_u64(9);

        let m = vec![
            vec![two, three, four],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        for i in 0..rows(&m) {
            let mut shadow = make_identity(columns(&m));
            let res = eliminate(&m, i, &mut shadow);
            if i > 0 {
                assert!(res.is_none());
                continue;
            } else {
                assert!(res.is_some());
            }

            assert_eq!(
                1,
                res.unwrap()
                    .iter()
                    .filter(|&row| row[i] != Fr::zero())
                    .count()
            );
        }
    }
    #[test]
    fn test_upper_triangular() {
        //        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);
        let seven = Fr::from_u64(7);
        let eight = Fr::from_u64(8);
        //        let nine = Fr::from_u64(9);

        let m = vec![
            vec![two, three, four],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        let mut shadow = make_identity(columns(&m));
        let _res = upper_triangular(&m, &mut shadow);

        // Actually assert things.
    }

    #[test]
    fn test_reduce_to_identity() {
        //        let one = Fr::from_u64(1);
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let four = Fr::from_u64(4);
        let five = Fr::from_u64(5);
        let six = Fr::from_u64(6);
        let seven = Fr::from_u64(7);
        let eight = Fr::from_u64(8);
        //        let nine = Fr::from_u64(9);

        let m = vec![
            vec![two, three, four],
            vec![four, five, six],
            vec![seven, eight, eight],
        ];

        let mut shadow = make_identity(columns(&m));
        let ut = upper_triangular(&m, &mut shadow);

        let res = ut
            .and_then(|x| reduce_to_identity(&x, &mut shadow))
            .unwrap();

        assert!(is_identity(&res));
        let prod = mat_mul(&m, &shadow).unwrap();

        assert!(is_identity(&prod));
    }
}
