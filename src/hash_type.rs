/// `HashType` provides support for domain separation tags.
/// For 128-bit security, we need to reserve one (~256-bit) field element per Poseidon permutation.
/// This element cannot be used for hash preimage data — but can be assigned a constant value designating
/// the hash function built on top of the underlying permutation.
///
/// `neptune` implements a variation of the domain separation tag scheme suggested in the updated Poseidon paper. This
/// allows for a variety of modes. This ensures that digest values produced using one hash function cannot be reused
/// where another is required.
///
/// Because `neptune` also supports a first-class notion of `Strength`, we include a mechanism for composing
/// `Strength` with `HashType` so that hashes with `Strength` other than `Standard` (currently only `Strengthened`)
/// may still express the full range of hash function types.
use crate::{field::PoseidonField, Arity, Strength};

#[derive(Clone, Debug, PartialEq)]
pub enum HashType<F: PoseidonField, A: Arity<F>> {
    MerkleTree,
    MerkleTreeSparse(u64),
    VariableLength,
    ConstantLength(usize),
    Encryption,
    Custom(CType<F, A>),
}

impl<F: PoseidonField, A: Arity<F>> HashType<F, A> {
    pub fn domain_tag(&self, strength: &Strength) -> F {
        let with_strength = |x: F| {
            let mut tmp = x;
            tmp.add_assign(&Self::strength_tag_component(strength));
            tmp
        };

        match self {
            // 2^arity - 1
            HashType::MerkleTree => with_strength(A::tag()),
            // bitmask
            HashType::MerkleTreeSparse(bitmask) => with_strength(F::from_u64(*bitmask)),
            // 2^64
            HashType::VariableLength => with_strength(pow2::<F>(64)),
            // length * 2^64
            // length must be greater than 0 and <= arity
            HashType::ConstantLength(length) => {
                assert!(*length as usize <= A::to_usize());
                assert!(*length as usize > 0);
                with_strength(x_pow2::<F>(*length as u64, 64))
            }
            // 2^32
            HashType::Encryption => with_strength(pow2::<F>(32)),
            // identifier * 2^40
            // NOTE: in order to leave room for future `Strength` tags,
            // we make identifier a multiple of 2^40 rather than 2^32.
            HashType::Custom(ref ctype) => ctype.domain_tag(&strength),
        }
    }

    fn strength_tag_component(strength: &Strength) -> F {
        let id = match strength {
            // Standard strength doesn't affect the base tag.
            Strength::Standard => 0,
            Strength::Strengthened => 1,
        };

        x_pow2::<F>(id, 32)
    }

    /// Some HashTypes require more testing so are not yet supported, since they are not yet needed.
    /// As and when needed, support can be added, along with tests to ensure the initial implementation
    /// is sound.
    pub fn is_supported(&self) -> bool {
        match self {
            HashType::MerkleTree => true,
            HashType::MerkleTreeSparse(_) => false,
            HashType::VariableLength => false,
            HashType::ConstantLength(_) => true,
            HashType::Encryption => true,
            HashType::Custom(_) => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CType<F: PoseidonField, A: Arity<F>> {
    Arbitrary(u64),
    _Phantom((F, A)),
}

impl<F: PoseidonField, A: Arity<F>> CType<F, A> {
    fn identifier(&self) -> u64 {
        match self {
            CType::Arbitrary(id) => *id,
            CType::_Phantom(_) => panic!("_Phantom is not a real custom tag type."),
        }
    }

    fn domain_tag(&self, _strength: &Strength) -> F {
        x_pow2::<F>(self.identifier(), 32)
    }
}

/// pow2(n) = 2^n
fn pow2<F: PoseidonField>(n: u64) -> F {
    F::from_u64(2).pow(n)
}

/// x_pow2(x, n) = x * 2^n
fn x_pow2<F: PoseidonField>(coeff: u64, n: u64) -> F {
    let mut tmp = pow2::<F>(n);
    tmp.mul_assign(&F::from_u64(coeff));
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Strength;
    use bellperson::bls::Fr;
    use fff::PrimeField;
    use generic_array::typenum::{U15, U8};
    use std::collections::HashSet;

    #[test]
    fn test_domain_tags() {
        let merkle_standard = HashType::MerkleTree::<Fr, U8>.domain_tag(&Strength::Standard);
        let expected_merkle_standard = Fr::from_le_u64s([
            0x00000000000000ff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ])
        .unwrap();

        assert_eq!(expected_merkle_standard, merkle_standard);

        let merkle_strengthened =
            HashType::MerkleTree::<Fr, U8>.domain_tag(&Strength::Strengthened);
        let expected_merkle_strengthened = Fr::from_le_u64s([
            0x00000001000000ff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ])
        .unwrap();
        assert_eq!(expected_merkle_strengthened, merkle_strengthened,);

        // TODO: tests for
        // MerkleTreeSparse(u64),
        // VariableLength,
        // Custom(CType<Fr, A>),

        let mut all_tags = Vec::new();

        for length in 1..15 {
            let constant_standard =
                HashType::ConstantLength::<Fr, U15>(length).domain_tag(&Strength::Standard);

            all_tags.push(constant_standard);

            if length <= 8 {
                let constant_standard_alt_arity =
                    HashType::ConstantLength::<Fr, U8>(length).domain_tag(&Strength::Standard);

                // Constant-length tag is independent of arity.
                assert_eq!(constant_standard, constant_standard_alt_arity);
            }

            assert_eq!(
                constant_standard,
                Fr::from_le_u64s([
                    0x0000000000000000,
                    length as u64,
                    0x0000000000000000,
                    0x0000000000000000
                ])
                .unwrap()
            );
        }

        for length in 1..15 {
            let constant_strengthened =
                HashType::ConstantLength::<Fr, U15>(length).domain_tag(&Strength::Strengthened);

            all_tags.push(constant_strengthened);

            if length <= 8 {
                let constant_strenghtened_alt_arity =
                    HashType::ConstantLength::<Fr, U8>(length).domain_tag(&Strength::Strengthened);

                // Constant-length tag is independent of arity.
                assert_eq!(constant_strengthened, constant_strenghtened_alt_arity);
            }

            assert_eq!(
                constant_strengthened,
                Fr::from_le_u64s([
                    0x0000000100000000,
                    length as u64,
                    0x0000000000000000,
                    0x0000000000000000
                ])
                .unwrap()
            );
        }

        let encryption_standard = HashType::Encryption::<Fr, U8>.domain_tag(&Strength::Standard);
        let expected_encryption_standard = Fr::from_le_u64s([
            0x0000000100000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ])
        .unwrap();
        assert_eq!(expected_encryption_standard, encryption_standard,);

        let encryption_strengthened =
            HashType::Encryption::<Fr, U8>.domain_tag(&Strength::Strengthened);
        let expected_encryption_strengthened = Fr::from_le_u64s([
            0x0000000200000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ])
        .unwrap();
        assert_eq!(expected_encryption_strengthened, encryption_strengthened);

        all_tags.extend(&[
            expected_merkle_standard,
            expected_merkle_strengthened,
            expected_encryption_standard,
            expected_encryption_strengthened,
        ]);

        let mut all_tags_set = HashSet::new();
        all_tags.iter().for_each(|x| {
            let _ = all_tags_set.insert(x.into_repr().0);
        });

        // Cardinality of set and vector are the same,
        // hence no tag is duplicated.
        assert_eq!(all_tags.len(), all_tags_set.len());
    }
}
