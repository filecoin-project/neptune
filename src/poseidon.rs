use crate::field::PoseidonField;
use crate::hash_type::HashType;
use crate::matrix::Matrix;
use crate::mds::{create_mds_matrices, factor_to_sparse_matrixes, MdsMatrices, SparseMatrix};
use crate::poseidon_alt::{hash_correct, hash_optimized_dynamic};
use crate::preprocessing::compress_round_constants;
use crate::{quintic_s_box, BatchHasher, Strength, DEFAULT_STRENGTH};
use crate::{round_constants, round_numbers, Error};
use bellperson::bls::Fr;
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;
use typenum::*;

/// Available arities for the Poseidon hasher.
pub trait Arity<T>: ArrayLength<T> {
    /// Must be Arity + 1.
    type ConstantsSize: ArrayLength<T>;

    fn tag() -> T;
}

macro_rules! impl_arity {
    ($($a:ty => $b:ty),*) => {
        $(
            impl<F: PoseidonField> Arity<F> for $a {
                type ConstantsSize = $b;

                fn tag() -> F {
                    F::from_u64((1 << Self::to_usize()) - 1)
                }
            }
        )*
    };
}

// Dummy implementation to allow for an "optional" argument.
impl<F: PoseidonField> Arity<F> for U0 {
    type ConstantsSize = U0;

    fn tag() -> F {
        unreachable!("dummy implementation for U0, should not be called")
    }
}

impl_arity!(
    U2 => U3,
    U3 => U4,
    U4 => U5,
    U5 => U6,
    U6 => U7,
    U7 => U8,
    U8 => U9,
    U9 => U10,
    U10 => U11,
    U11 => U12,
    U12 => U13,
    U13 => U14,
    U14 => U15,
    U15 => U16,
    U16 => U17,
    U17 => U18,
    U18 => U19,
    U19 => U20,
    U20 => U21,
    U21 => U22,
    U22 => U23,
    U23 => U24,
    U24 => U25,
    U25 => U26,
    U26 => U27,
    U27 => U28,
    U28 => U29,
    U29 => U30,
    U30 => U31,
    U31 => U32,
    U32 => U33,
    U33 => U34,
    U34 => U35,
    U35 => U36,
    U36 => U37
);

/// The `Poseidon` structure will accept a number of inputs equal to the arity.
#[derive(Debug, Clone, PartialEq)]
pub struct Poseidon<'a, F, A = U2>
where
    F: PoseidonField,
    A: Arity<F>,
{
    pub(crate) constants_offset: usize,
    pub(crate) current_round: usize, // Used in static optimization only for now.
    /// the elements to permute
    pub elements: GenericArray<F, A::ConstantsSize>,
    pos: usize,
    pub(crate) constants: &'a PoseidonConstants<F, A>,
    _f: PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseidonConstants<F, A>
where
    F: PoseidonField,
    A: Arity<F>,
{
    pub mds_matrices: MdsMatrices<F>,
    pub round_constants: Vec<F>,
    pub compressed_round_constants: Vec<F>,
    pub pre_sparse_matrix: Matrix<F>,
    pub sparse_matrixes: Vec<SparseMatrix<F>>,
    pub strength: Strength,
    /// The domain tag is the first element of a Poseidon permutation.
    /// This extra element is necessary for 128-bit security.
    pub domain_tag: F,
    pub full_rounds: usize,
    pub half_full_rounds: usize,
    pub partial_rounds: usize,
    pub hash_type: HashType<F, A>,
    _a: PhantomData<A>,
}

#[derive(Debug, PartialEq)]
pub enum HashMode {
    // The initial and correct version of the algorithm. We should preserve the ability to hash this way for reference
    // and to preserve confidence in our tests along thew way.
    Correct,
    // This mode is meant to be mostly synchronized with `Correct` but may reduce or simplify the work performed by the
    // algorithm, if not the code implementing. Its purpose is for use during refactoring/development.
    OptimizedDynamic,
    // Consumes statically pre-processed constants for simplest operation.
    OptimizedStatic,
}
use HashMode::{Correct, OptimizedDynamic, OptimizedStatic};

pub const DEFAULT_HASH_MODE: HashMode = OptimizedStatic;

impl<'a, F, A> PoseidonConstants<F, A>
where
    F: PoseidonField,
    A: Arity<F>,
{
    pub fn new() -> Self {
        Self::new_with_strength(DEFAULT_STRENGTH)
    }

    /// `new_constant_length` creates constants for hashing a constant-sized preimage which is <= the max
    /// supported by the permutation width.
    pub fn new_constant_length(length: usize) -> Self {
        let arity = A::to_usize();
        assert!(length <= arity);
        Self::new_with_strength_and_type(DEFAULT_STRENGTH, HashType::ConstantLength(length))
    }

    pub fn with_length(&self, length: usize) -> Self {
        let arity = A::to_usize();
        assert!(length <= arity);

        let hash_type = match self.hash_type {
            HashType::ConstantLength(_) => HashType::ConstantLength(length),
            _ => panic!("cannot set constant length of hash without type ConstantLength."),
        };

        let domain_tag = hash_type.domain_tag(&self.strength);

        Self {
            hash_type,
            domain_tag,
            ..self.clone()
        }
    }

    pub fn new_with_strength(strength: Strength) -> Self {
        Self::new_with_strength_and_type(strength, HashType::MerkleTree)
    }

    pub fn new_with_strength_and_type(strength: Strength, hash_type: HashType<F, A>) -> Self {
        assert!(hash_type.is_supported());
        let arity = A::to_usize();
        let width = arity + 1;

        let mds_matrices = create_mds_matrices(width);

        let (full_rounds, partial_rounds) = round_numbers(arity, &strength);
        let half_full_rounds = full_rounds / 2;
        let round_constants = round_constants(arity, &strength);
        let compressed_round_constants = compress_round_constants(
            width,
            full_rounds,
            partial_rounds,
            &round_constants,
            &mds_matrices,
            partial_rounds,
        );

        let (pre_sparse_matrix, sparse_matrixes) =
            factor_to_sparse_matrixes(mds_matrices.m.clone(), partial_rounds);

        // Ensure we have enough constants for the sbox rounds
        assert!(
            width * (full_rounds + partial_rounds) <= round_constants.len(),
            "Not enough round constants"
        );

        assert_eq!(
            full_rounds * width + partial_rounds,
            compressed_round_constants.len()
        );

        Self {
            mds_matrices,
            round_constants,
            compressed_round_constants,
            pre_sparse_matrix,
            sparse_matrixes,
            strength,
            domain_tag: hash_type.domain_tag(&strength),
            full_rounds,
            half_full_rounds,
            partial_rounds,
            hash_type,
            _a: PhantomData::<A>,
        }
    }

    /// Returns the width.
    #[inline]
    pub fn arity(&self) -> usize {
        A::to_usize()
    }

    /// Returns the width.
    #[inline]
    pub fn width(&self) -> usize {
        A::ConstantsSize::to_usize()
    }
}

impl<F, A> Default for PoseidonConstants<F, A>
where
    F: PoseidonField,
    A: Arity<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, F, A> Poseidon<'a, F, A>
where
    F: PoseidonField,
    A: Arity<F>,
{
    pub fn new(constants: &'a PoseidonConstants<F, A>) -> Self {
        let elements = GenericArray::generate(|i| {
            if i == 0 {
                constants.domain_tag
            } else {
                F::zero()
            }
        });
        Poseidon {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: 1,
            constants,
            _f: PhantomData::<F>,
        }
    }

    pub fn new_with_preimage(preimage: &[F], constants: &'a PoseidonConstants<F, A>) -> Self {
        let elements = match constants.hash_type {
            HashType::ConstantLength(constant_len) => {
                assert_eq!(constant_len, preimage.len(), "Invalid preimage size");

                GenericArray::generate(|i| {
                    if i == 0 {
                        constants.domain_tag
                    } else if i > preimage.len() {
                        F::zero()
                    } else {
                        preimage[i - 1]
                    }
                })
            }
            HashType::VariableLength => panic!("variable-length hashes are not yet supported."),
            _ => {
                assert_eq!(preimage.len(), A::to_usize(), "Invalid preimage size");

                GenericArray::generate(|i| {
                    if i == 0 {
                        constants.domain_tag
                    } else {
                        preimage[i - 1]
                    }
                })
            }
        };
        let width = preimage.len();

        Poseidon {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: width,
            constants,
            _f: PhantomData::<F>,
        }
    }

    /// Replace the elements with the provided optional items.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is bigger than the arity.
    pub fn set_preimage(&mut self, preimage: &[F]) {
        self.reset();
        self.elements[1..].copy_from_slice(&preimage);
        self.pos = self.elements.len();
    }

    /// Restore the initial state
    pub fn reset(&mut self) {
        self.constants_offset = 0;
        self.current_round = 0;
        self.elements[1..]
            .iter_mut()
            .for_each(|l| *l = F::from_u64(0u64));
        self.elements[0] = self.constants.domain_tag;
        self.pos = 1;
    }

    /// The returned `usize` represents the element position (within arity) for the input operation
    pub fn input(&mut self, element: F) -> Result<usize, Error> {
        // Cannot input more elements than the defined arity
        if self.pos >= self.constants.width() {
            return Err(Error::FullBuffer);
        }

        // Set current element, and increase the pointer
        self.elements[self.pos] = element;
        self.pos += 1;

        Ok(self.pos - 1)
    }

    pub fn hash_in_mode(&mut self, mode: HashMode) -> F {
        self.apply_padding();
        match mode {
            Correct => hash_correct(self),
            OptimizedDynamic => hash_optimized_dynamic(self),
            OptimizedStatic => self.hash_optimized_static(),
        }
    }

    pub fn hash(&mut self) -> F {
        self.hash_in_mode(DEFAULT_HASH_MODE)
    }

    fn apply_padding(&mut self) {
        match self.constants.hash_type {
            HashType::ConstantLength(l) => {
                assert_eq!(
                    self.pos, l,
                    "preimage length does not match constant length required for hash"
                );
                // There is nothing to do here, but only because the state elements were
                // initialized to zero, and that is what we need to pad with.
            }
            HashType::VariableLength => todo!(),
            _ => (),
        }
    }

    pub fn hash_optimized_static(&mut self) -> F {
        // The first full round should use the initial constants.
        self.add_round_constants();

        for _ in 0..self.constants.half_full_rounds {
            self.full_round(false);
        }

        for _ in 0..self.constants.partial_rounds {
            self.partial_round();
        }

        // All but last full round.
        for _ in 1..self.constants.half_full_rounds {
            self.full_round(false);
        }
        self.full_round(true);

        assert_eq!(
            self.constants_offset,
            self.constants.compressed_round_constants.len(),
            "Constants consumed ({}) must equal preprocessed constants provided ({}).",
            self.constants_offset,
            self.constants.compressed_round_constants.len()
        );

        self.elements[1]
    }

    fn full_round(&mut self, last_round: bool) {
        let to_take = self.elements.len();
        let post_round_keys = self
            .constants
            .compressed_round_constants
            .iter()
            .skip(self.constants_offset)
            .take(to_take);

        if !last_round {
            let needed = self.constants_offset + to_take;
            assert!(
                needed <= self.constants.compressed_round_constants.len(),
                "Not enough preprocessed round constants ({}), need {}.",
                self.constants.compressed_round_constants.len(),
                needed
            );
        }
        self.elements
            .iter_mut()
            .zip(post_round_keys)
            .for_each(|(l, post)| {
                // Be explicit that no round key is added after last round of S-boxes.
                let post_key = if last_round {
                    panic!(
                        "Trying to skip last full round, but there is a key here! ({:?})",
                        post
                    );
                } else {
                    Some(post)
                };
                quintic_s_box(l, None, post_key);
            });
        // We need this because post_round_keys will have been empty, so it didn't happen in the for_each. :(
        if last_round {
            self.elements
                .iter_mut()
                .for_each(|l| quintic_s_box(l, None, None));
        } else {
            self.constants_offset += self.elements.len();
        }
        self.round_product_mds();
    }

    /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first (arity tag) poseidon leaf.
    fn partial_round(&mut self) {
        let post_round_key = self.constants.compressed_round_constants[self.constants_offset];

        // Apply the quintic S-Box to the first element
        quintic_s_box(&mut self.elements[0], None, Some(&post_round_key));
        self.constants_offset += 1;

        self.round_product_mds();
    }

    fn add_round_constants(&mut self) {
        for (element, round_constant) in self.elements.iter_mut().zip(
            self.constants
                .compressed_round_constants
                .iter()
                .skip(self.constants_offset),
        ) {
            element.add_assign(round_constant);
        }
        self.constants_offset += self.elements.len();
    }

    /// Set the provided elements with the result of the product between the elements and the appropriate
    /// MDS matrix.
    #[allow(clippy::collapsible_else_if)]
    fn round_product_mds(&mut self) {
        let full_half = self.constants.half_full_rounds;
        let sparse_offset = full_half - 1;
        if self.current_round == sparse_offset {
            self.product_mds_with_matrix(&self.constants.pre_sparse_matrix);
        } else {
            if (self.current_round > sparse_offset)
                && (self.current_round < full_half + self.constants.partial_rounds)
            {
                let index = self.current_round - sparse_offset - 1;
                let sparse_matrix = &self.constants.sparse_matrixes[index];

                self.product_mds_with_sparse_matrix(&sparse_matrix);
            } else {
                self.product_mds();
            }
        };

        self.current_round += 1;
    }

    /// Set the provided elements with the result of the product between the elements and the constant
    /// MDS matrix.
    pub(crate) fn product_mds(&mut self) {
        self.product_mds_with_matrix(&self.constants.mds_matrices.m);
    }

    /// NOTE: This calculates a vector-matrix product (`elements * matrix`) rather than the
    /// expected matrix-vector `(matrix * elements)`. This is a performance optimization which
    /// exploits the fact that our MDS matrices are symmetric by construction.
    #[allow(clippy::ptr_arg)]
    pub(crate) fn product_mds_with_matrix(&mut self, matrix: &Matrix<F>) {
        let mut result = GenericArray::<F, A::ConstantsSize>::generate(|_| F::zero());

        for (j, val) in result.iter_mut().enumerate() {
            for (i, row) in matrix.iter().enumerate() {
                let mut tmp = row[j];
                tmp.mul_assign(&self.elements[i]);
                val.add_assign(&tmp);
            }
        }

        let _ = std::mem::replace(&mut self.elements, result);
    }

    // Sparse matrix in this context means one of the form, M''.
    fn product_mds_with_sparse_matrix(&mut self, sparse_matrix: &SparseMatrix<F>) {
        let mut result = GenericArray::<F, A::ConstantsSize>::generate(|_| F::zero());

        // First column is dense.
        for (i, val) in sparse_matrix.w_hat.iter().enumerate() {
            let mut tmp = *val;
            tmp.mul_assign(&self.elements[i]);
            result[0].add_assign(&tmp);
        }

        for (j, val) in result.iter_mut().enumerate().skip(1) {
            // Except for first row/column, diagonals are one.
            val.add_assign(&self.elements[j]);

            // First row is dense.
            let mut tmp = sparse_matrix.v_rest[j - 1];
            tmp.mul_assign(&self.elements[0]);
            val.add_assign(&tmp);
        }

        let _ = std::mem::replace(&mut self.elements, result);
    }

    fn debug(&self, msg: &str) {
        dbg!(msg, &self.constants_offset, &self.elements);
    }
}

#[derive(Debug)]
pub struct SimplePoseidonBatchHasher<A>
where
    A: Arity<Fr>,
{
    constants: PoseidonConstants<Fr, A>,
    max_batch_size: usize,
}

impl<A> SimplePoseidonBatchHasher<A>
where
    A: Arity<Fr>,
{
    pub(crate) fn new(max_batch_size: usize) -> Self {
        Self::new_with_strength(DEFAULT_STRENGTH, max_batch_size)
    }

    pub(crate) fn new_with_strength(strength: Strength, max_batch_size: usize) -> Self {
        Self {
            constants: PoseidonConstants::<Fr, A>::new_with_strength(strength),
            max_batch_size,
        }
    }
}
impl<A> BatchHasher<A> for SimplePoseidonBatchHasher<A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        Ok(preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &self.constants).hash())
            .collect())
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use bellperson::bls::Fr;
    use fff::Field;
    use generic_array::typenum;

    #[test]
    fn reset() {
        let test_arity = 2;
        let preimage = vec![<Fr as Field>::one(); test_arity];
        let constants = PoseidonConstants::new();
        let mut h = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants);
        h.hash();
        h.reset();

        let default = Poseidon::<Fr, U2>::new(&constants);
        assert_eq!(default.pos, h.pos);
        assert_eq!(default.elements, h.elements);
        assert_eq!(default.constants_offset, h.constants_offset);
    }

    #[test]
    fn hash_det() {
        let test_arity = 2;
        let mut preimage = vec![<Fr as Field>::zero(); test_arity];
        let constants = PoseidonConstants::new();
        preimage[0] = <Fr as Field>::one();

        let mut h = Poseidon::<Fr, U2>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_arity_3() {
        let mut preimage: [Fr; 3] = [<Fr as Field>::zero(); 3];
        let constants = PoseidonConstants::new();
        preimage[0] = <Fr as Field>::one();

        let mut h = Poseidon::<Fr, typenum::U3>::new_with_preimage(&preimage, &constants);

        let mut h2 = h.clone();
        let result = h.hash();

        assert_eq!(result, h2.hash());
    }

    #[test]
    fn hash_values() {
        hash_values_cases(Strength::Standard);
        hash_values_cases(Strength::Strengthened);
    }

    fn hash_values_cases(strength: Strength) {
        hash_values_aux::<typenum::U2>(strength);
        hash_values_aux::<typenum::U4>(strength);
        hash_values_aux::<typenum::U8>(strength);
        hash_values_aux::<typenum::U11>(strength);
        hash_values_aux::<typenum::U16>(strength);
        hash_values_aux::<typenum::U24>(strength);
        hash_values_aux::<typenum::U36>(strength);
    }

    /// Simple test vectors to ensure results don't change unintentionally in development.
    fn hash_values_aux<A>(strength: Strength)
    where
        A: Arity<Fr>,
    {
        let constants = PoseidonConstants::<Fr, A>::new_with_strength(strength);
        let mut p = Poseidon::<Fr, A>::new(&constants);
        let mut p2 = Poseidon::<Fr, A>::new(&constants);
        let mut p3 = Poseidon::<Fr, A>::new(&constants);
        let mut p4 = Poseidon::<Fr, A>::new(&constants);

        let test_arity = constants.arity();
        for n in 0..test_arity {
            let scalar = Fr::from_u64(n as u64);
            p.input(scalar).unwrap();
            p2.input(scalar).unwrap();
            p3.input(scalar).unwrap();
            p4.input(scalar).unwrap();
        }

        let digest = p.hash();
        let digest2 = p2.hash_in_mode(Correct);
        let digest3 = p3.hash_in_mode(OptimizedStatic);
        let digest4 = p4.hash_in_mode(OptimizedDynamic);
        assert_eq!(digest, digest2);
        assert_eq!(digest, digest3);
        assert_eq!(digest, digest4);

        let expected = match strength {
            Strength::Standard => {
                // Currently secure round constants.
                match test_arity {
                    2 => Fr::from_le_u64s([
                        0x2e203c369a02e7ff,
                        0xa6fba9339d05a69d,
                        0x739e0fd902efe161,
                        0x396508d75e76a56b,
                    ])
                    .unwrap(),
                    4 => Fr::from_le_u64s([
                        0x019814ff6662075d,
                        0xfb6b4605bf1327ec,
                        0x00db3c6579229399,
                        0x58a54b10a9e5848a,
                    ])
                    .unwrap(),
                    8 => Fr::from_le_u64s([
                        0x2a9934f56d38a5e6,
                        0x4b682e9d9cc4aed9,
                        0x1201004211677077,
                        0x2394611da3a5de55,
                    ])
                    .unwrap(),
                    11 => Fr::from_le_u64s([
                        0xcee3bbc32b693163,
                        0x09f3dcd8ccb08fc1,
                        0x6ca537e232ebe87a,
                        0x0c0fc1b2e5227f28,
                    ])
                    .unwrap(),
                    16 => Fr::from_le_u64s([
                        0x1291c74060266d37,
                        0x5b8dbc6d30680a6f,
                        0xc1c2fb5a6f871e63,
                        0x2d3ae2663381ae8a,
                    ])
                    .unwrap(),
                    24 => Fr::from_le_u64s([
                        0xd7ef3569f585b321,
                        0xc3e779f6468815b1,
                        0x066f39bf783f3d9f,
                        0x63beb8831f11ae15,
                    ])
                    .unwrap(),
                    36 => Fr::from_le_u64s([
                        0x4473606dfa4e8140,
                        0x75cd368df8a8ac3c,
                        0x540a30e03c10bbaa,
                        0x699303082a6e5d5f,
                    ])
                    .unwrap(),
                    _ => {
                        dbg!(digest, test_arity);
                        panic!("Arity lacks test vector: {}", test_arity)
                    }
                }
            }
            Strength::Strengthened =>
            // Strengthened round constants.
            {
                match test_arity {
                    2 => Fr::from_le_u64s([
                        0x3abccd9afc5729b1,
                        0x31662bb49883a7dc,
                        0x2a0ae894f8500373,
                        0x5f3027eb2ef4f4b8,
                    ])
                    .unwrap(),
                    4 => Fr::from_le_u64s([
                        0x3ff99d0422e647ee,
                        0xad9fc9ebbb1515e1,
                        0x8f57e5ab121004ce,
                        0x40223b87a6bd4508,
                    ])
                    .unwrap(),
                    8 => Fr::from_le_u64s([
                        0xfffbca3d9ffcda00,
                        0x7e4929e97170e2ae,
                        0xfdbbbd4b1b984b9b,
                        0x1367e3ced3e2edcb,
                    ])
                    .unwrap(),
                    11 => Fr::from_le_u64s([
                        0x29d77677fef45927,
                        0x39062662a7311a7a,
                        0xa8650443f7bf09c1,
                        0x7344835ba9059929,
                    ])
                    .unwrap(),
                    16 => Fr::from_le_u64s([
                        0x48f16b2a7fa48951,
                        0xbf999529774a192f,
                        0x273664a5bf751815,
                        0x6f53127e18f90e54,
                    ])
                    .unwrap(),
                    24 => Fr::from_le_u64s([
                        0xce136f2a6675f44b,
                        0x0bf949d57c82de03,
                        0xeab0b00318558589,
                        0x70015999f995274e,
                    ])
                    .unwrap(),
                    36 => Fr::from_le_u64s([
                        0x80098c6336781a9a,
                        0x591e29eb290a5b8e,
                        0xd26ff2e8c5dd73e4,
                        0x41d1adc5ece688c0,
                    ])
                    .unwrap(),
                    _ => {
                        dbg!(digest, test_arity);
                        panic!("Arity lacks test vector: {}", test_arity)
                    }
                }
            }
        };
        dbg!(test_arity);
        assert_eq!(expected, digest);
    }

    #[test]
    fn hash_compare_optimized() {
        let constants = PoseidonConstants::<Fr, U2>::new();
        let mut p = Poseidon::<Fr, U2>::new(&constants);
        let test_arity = constants.arity();
        for n in 0..test_arity {
            let scalar = Fr::from_u64(n as u64);
            p.input(scalar).unwrap();
        }
        let mut p2 = p.clone();
        let mut p3 = p.clone();

        let digest_correct = p.hash_in_mode(Correct);

        let digest_optimized_dynamic = p2.hash_in_mode(OptimizedDynamic);
        let digest_optimized_static = p3.hash_in_mode(OptimizedStatic);

        assert_eq!(digest_correct, digest_optimized_dynamic);
        assert_eq!(digest_correct, digest_optimized_static);
    }

    #[test]
    fn default_is_standard() {
        let default_constants = PoseidonConstants::<Fr, U8>::new();
        let standard_constants = PoseidonConstants::<Fr, U8>::new_with_strength(Strength::Standard);

        assert_eq!(
            standard_constants.partial_rounds,
            default_constants.partial_rounds
        );
    }
}
