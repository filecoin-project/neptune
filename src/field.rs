/// This module creates a trait `PoseidonField` and implements it for all BLS12-381 and Pasta scalar
/// fields. This trait rebundles the prime field functionality required by `neptune` into a single
/// interface. This rebundling is necessary because the BLS12-381 and Pasta scalar fields types
/// implement different `Field` and `PrimeField` traits, i.e. BLS12-381 scalar fields implement
/// `fff` traits, whereas the Pasta scalar fields implement `ff` traits.
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, SubAssign};

use blstrs::Scalar;
use paired::bls12_381::Fr;
use pasta_curves::{Fp, Fq};

pub trait PoseidonField: Clone + Copy + Debug + Eq {
    const NUM_BITS: u32;

    fn zero() -> Self;
    fn one() -> Self;
    fn from_u64(x: u64) -> Self;
    #[cfg(test)]
    fn from_le_u64s(u64s: [u64; 4]) -> Option<Self>;
    fn from_be_bytes(be_bytes: &[u8]) -> Option<Self>;
    fn add_assign(&mut self, other: &Self);
    fn sub_assign(&mut self, other: &Self);
    fn mul_assign(&mut self, other: &Self);
    fn square(&mut self);
    fn inverse(&self) -> Option<Self>;
    fn pow(&self, n: u64) -> Self;
}

impl PoseidonField for Fr {
    const NUM_BITS: u32 = <Self as fff::PrimeField>::NUM_BITS;

    fn zero() -> Self {
        <Self as fff::Field>::zero()
    }

    fn one() -> Self {
        <Self as fff::Field>::one()
    }

    fn from_u64(x: u64) -> Self {
        <Self as fff::PrimeField>::from_repr(<Self as fff::PrimeField>::Repr::from(x)).unwrap()
    }

    #[cfg(test)]
    fn from_le_u64s(u64s: [u64; 4]) -> Option<Self> {
        use paired::bls12_381::FrRepr;
        <Self as fff::PrimeField>::from_repr(FrRepr(u64s)).ok()
    }

    fn from_be_bytes(be_bytes: &[u8]) -> Option<Self> {
        let mut fr_repr = <<Self as fff::PrimeField>::Repr as Default>::default();
        // Each `u64` limb is 8 bytes.
        assert_eq!(be_bytes.len(), fr_repr.as_ref().len() * 8);
        <<Self as fff::PrimeField>::Repr as fff::PrimeFieldRepr>::read_be(&mut fr_repr, be_bytes)
            .unwrap();
        <Self as fff::PrimeField>::from_repr(fr_repr).ok()
    }

    fn add_assign(&mut self, other: &Self) {
        <Self as fff::Field>::add_assign(self, other);
    }

    fn sub_assign(&mut self, other: &Self) {
        <Self as fff::Field>::sub_assign(self, other);
    }

    fn mul_assign(&mut self, other: &Self) {
        <Self as fff::Field>::mul_assign(self, other);
    }

    fn square(&mut self) {
        <Self as fff::Field>::square(self);
    }

    fn inverse(&self) -> Option<Self> {
        <Self as fff::Field>::inverse(self)
    }

    fn pow(&self, n: u64) -> Self {
        <Self as fff::Field>::pow(self, &[n])
    }
}

impl PoseidonField for Scalar {
    const NUM_BITS: u32 = <Self as fff::PrimeField>::NUM_BITS;

    fn zero() -> Self {
        <Self as fff::Field>::zero()
    }

    fn one() -> Self {
        <Self as fff::Field>::one()
    }

    fn from_u64(x: u64) -> Self {
        <Self as fff::PrimeField>::from_repr(<Self as fff::PrimeField>::Repr::from(x)).unwrap()
    }

    #[cfg(test)]
    fn from_le_u64s(u64s: [u64; 4]) -> Option<Self> {
        use blstrs::ScalarRepr;
        <Self as fff::PrimeField>::from_repr(ScalarRepr(u64s)).ok()
    }

    fn from_be_bytes(be_bytes: &[u8]) -> Option<Self> {
        let mut fr_repr = <<Self as fff::PrimeField>::Repr as Default>::default();
        // Each `u64` limb is 8 bytes.
        assert_eq!(be_bytes.len(), fr_repr.as_ref().len() * 8);
        <<Self as fff::PrimeField>::Repr as fff::PrimeFieldRepr>::read_be(&mut fr_repr, be_bytes)
            .unwrap();
        <Self as fff::PrimeField>::from_repr(fr_repr).ok()
    }

    fn add_assign(&mut self, other: &Self) {
        <Self as fff::Field>::add_assign(self, other);
    }

    fn sub_assign(&mut self, other: &Self) {
        <Self as fff::Field>::sub_assign(self, other);
    }

    fn mul_assign(&mut self, other: &Self) {
        <Self as fff::Field>::mul_assign(self, other);
    }

    fn square(&mut self) {
        <Self as fff::Field>::square(self);
    }

    fn inverse(&self) -> Option<Self> {
        <Self as fff::Field>::inverse(self)
    }

    fn pow(&self, n: u64) -> Self {
        <Self as fff::Field>::pow(self, &[n])
    }
}

impl PoseidonField for Fp {
    const NUM_BITS: u32 = <Self as ff::PrimeField>::NUM_BITS;

    fn zero() -> Self {
        <Self as ff::Field>::zero()
    }

    fn one() -> Self {
        <Self as ff::Field>::one()
    }

    fn from_u64(x: u64) -> Self {
        <Self as From<u64>>::from(x)
    }

    #[cfg(test)]
    fn from_le_u64s(u64s: [u64; 4]) -> Option<Self> {
        let mut le_bytes = <<Self as ff::PrimeField>::Repr as Default>::default();
        for (i, int) in u64s.iter().enumerate() {
            let start = 8 * i;
            le_bytes[start..start + 8].copy_from_slice(&int.to_le_bytes());
        }
        <Self as ff::PrimeField>::from_repr(le_bytes)
    }

    fn from_be_bytes(be_bytes: &[u8]) -> Option<Self> {
        // The Pasta scalar fields have reprs which are an array of 32 little-endian bytes (whereas
        // BLS12-381 scalar fields have reprs which are arrays of 4 `u64` limbs). Thus, we must
        // convert the big-endian bytes slice `be_bytes` into a little-endian array.
        let mut le_bytes = <<Self as ff::PrimeField>::Repr as Default>::default();
        assert_eq!(be_bytes.len(), le_bytes.len());
        be_bytes
            .iter()
            .rev()
            .zip(le_bytes.iter_mut())
            .for_each(|(src, dst)| {
                *dst = *src;
            });
        <Self as ff::PrimeField>::from_repr(le_bytes)
    }

    fn add_assign(&mut self, other: &Self) {
        <Self as AddAssign<&Self>>::add_assign(self, other);
    }

    fn sub_assign(&mut self, other: &Self) {
        <Self as SubAssign<&Self>>::sub_assign(self, other);
    }

    fn mul_assign(&mut self, other: &Self) {
        <Self as MulAssign<&Self>>::mul_assign(self, other);
    }

    fn square(&mut self) {
        *self = <Self as ff::Field>::square(self);
    }

    fn inverse(&self) -> Option<Self> {
        let inv = <Self as ff::Field>::invert(self);
        if inv.is_some().into() {
            Some(inv.unwrap())
        } else {
            None
        }
    }

    fn pow(&self, n: u64) -> Self {
        <Self as ff::Field>::pow_vartime(self, &[n])
    }
}

impl PoseidonField for Fq {
    const NUM_BITS: u32 = <Self as ff::PrimeField>::NUM_BITS;

    fn zero() -> Self {
        <Self as ff::Field>::zero()
    }

    fn one() -> Self {
        <Self as ff::Field>::one()
    }

    fn from_u64(x: u64) -> Self {
        <Self as From<u64>>::from(x)
    }

    #[cfg(test)]
    fn from_le_u64s(u64s: [u64; 4]) -> Option<Self> {
        let mut le_bytes = <<Self as ff::PrimeField>::Repr as Default>::default();
        for (i, int) in u64s.iter().enumerate() {
            let start = 8 * i;
            le_bytes[start..start + 8].copy_from_slice(&int.to_le_bytes());
        }
        <Self as ff::PrimeField>::from_repr(le_bytes)
    }

    fn from_be_bytes(be_bytes: &[u8]) -> Option<Self> {
        // The Pasta scalar fields have reprs which are an array of 32 little-endian bytes (whereas
        // BLS12-381 scalar fields have reprs which are arrays of 4 `u64` limbs). Thus, we must
        // convert the big-endian bytes slice `be_bytes` into a little-endian array.
        let mut le_bytes = <<Self as ff::PrimeField>::Repr as Default>::default();
        assert_eq!(be_bytes.len(), le_bytes.len());
        be_bytes
            .iter()
            .rev()
            .zip(le_bytes.iter_mut())
            .for_each(|(src, dst)| {
                *dst = *src;
            });
        <Self as ff::PrimeField>::from_repr(le_bytes)
    }

    fn add_assign(&mut self, other: &Self) {
        <Self as AddAssign<&Self>>::add_assign(self, other);
    }

    fn sub_assign(&mut self, other: &Self) {
        <Self as SubAssign<&Self>>::sub_assign(self, other);
    }

    fn mul_assign(&mut self, other: &Self) {
        <Self as MulAssign<&Self>>::mul_assign(self, other);
    }

    fn square(&mut self) {
        *self = <Self as ff::Field>::square(self);
    }

    fn inverse(&self) -> Option<Self> {
        let inv = <Self as ff::Field>::invert(self);
        if inv.is_some().into() {
            Some(inv.unwrap())
        } else {
            None
        }
    }

    fn pow(&self, n: u64) -> Self {
        <Self as ff::Field>::pow_vartime(self, &[n])
    }
}
