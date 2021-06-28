use super::cl;
use crate::error::Error;
use crate::hash_type::HashType;
use crate::poseidon::PoseidonConstants;
use crate::{Arity, BatchHasher, Strength, DEFAULT_STRENGTH};
use bellperson::bls::{Fr, FrRepr};
use fff::{PrimeField, PrimeFieldDecodingError};
use generic_array::{typenum, ArrayLength, GenericArray};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use triton::FutharkContext;
use triton::{Array_u64_1d, Array_u64_2d, Array_u64_3d};
use typenum::{U11, U2, U8};

/// Convenience type aliases for opaque pointers from the generated Futhark bindings.
type P2State = triton::FutharkOpaqueP2State;
type P8State = triton::FutharkOpaqueP8State;
type P11State = triton::FutharkOpaqueP11State;

type S2State = triton::FutharkOpaqueS2State;
type S8State = triton::FutharkOpaqueS8State;
type S11State = triton::FutharkOpaqueS11State;

pub(crate) type T864MState = triton::FutharkOpaqueT864MState;

#[derive(Clone, Hash, Eq, PartialEq)]
struct FutharkContextPointer(*const FutharkContext);
unsafe impl Send for FutharkContextPointer {}
unsafe impl Sync for FutharkContextPointer {}

lazy_static! {
    static ref BATCH_HASHERS: Mutex<HashMap<FutharkContextPointer, usize>> =
        Mutex::new(HashMap::default());
}

/// Container to hold the state corresponding to each supported arity.
enum BatcherState {
    Arity2(P2State),
    Arity8(P8State),
    Arity11(P11State),
    Arity2s(S2State),
    Arity8s(S8State),
    Arity11s(S11State),
}

impl BatcherState {
    /// Create a new state for use in batch hashing preimages of `Arity` elements.
    /// State is an opaque pointer supplied to the corresponding GPU entry point when processing a batch.
    fn new<A: Arity<Fr>>(ctx: Arc<Mutex<FutharkContext>>) -> Result<Self, Error> {
        Self::new_with_strength::<A>(ctx, DEFAULT_STRENGTH)
    }
    fn new_with_strength<A: Arity<Fr>>(
        ctx: Arc<Mutex<FutharkContext>>,
        strength: Strength,
    ) -> Result<Self, Error> {
        let mut ctx = ctx.lock().unwrap();
        Ok(match A::to_usize() {
            size if size == 2 => init_hash2(&mut ctx, strength)?,
            size if size == 8 => init_hash8(&mut ctx, strength)?,
            size if size == 11 => init_hash11(&mut ctx, strength)?,
            _ => panic!("unsupported arity: {}", A::to_usize()),
        })
    }

    /// Hash a batch of N * `Arity` `Fr`s into N `Fr`s.
    fn hash<A: ArrayLength<Fr>>(
        &mut self,
        ctx: &mut FutharkContext,
        preimages: &[GenericArray<Fr, A>],
    ) -> Result<(Vec<Fr>, Self), Error>
    where
        A: Arity<Fr>,
    {
        match self {
            BatcherState::Arity2(state) => {
                let (res, state) = mbatch_hash2(ctx, state, preimages)?;
                Ok((res, BatcherState::Arity2(state)))
            }
            BatcherState::Arity8(state) => {
                let (res, state) = mbatch_hash8(ctx, state, preimages)?;
                Ok((res, BatcherState::Arity8(state)))
            }
            BatcherState::Arity11(state) => {
                let (res, state) = mbatch_hash11(ctx, state, preimages)?;
                Ok((res, BatcherState::Arity11(state)))
            }
            BatcherState::Arity2s(state) => {
                let (res, state) = mbatch_hash2s(ctx, state, preimages)?;
                Ok((res, BatcherState::Arity2s(state)))
            }
            BatcherState::Arity8s(state) => {
                let (res, state) = mbatch_hash8s(ctx, state, preimages)?;
                Ok((res, BatcherState::Arity8s(state)))
            }
            BatcherState::Arity11s(state) => {
                let (res, state) = mbatch_hash11s(ctx, state, preimages)?;
                Ok((res, BatcherState::Arity11s(state)))
            }
        }
    }
}

/// `GpuBatchHasher` implements `BatchHasher` and performs the batched hashing on GPU.
pub struct GpuBatchHasher<A> {
    ctx: Arc<Mutex<FutharkContext>>,
    state: BatcherState,
    /// If `tree_builder_state` is provided, use it to build the final 64MiB tree on the GPU with one call.
    tree_builder_state: Option<T864MState>,
    max_batch_size: usize,
    _a: PhantomData<A>,
}

impl<A> GpuBatchHasher<A>
where
    A: Arity<Fr>,
{
    /// Create a new `GpuBatchHasher` and initialize it with state corresponding with its `A`.
    pub(crate) fn new(
        ctx: Arc<Mutex<FutharkContext>>,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        Self::new_with_strength(ctx, DEFAULT_STRENGTH, max_batch_size)
    }

    /// Create a new `GpuBatchHasher` and initialize it with state corresponding with its `A`.
    pub(crate) fn new_with_strength(
        ctx: Arc<Mutex<FutharkContext>>,
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        let new = Self {
            ctx: Arc::clone(&ctx),
            state: BatcherState::new_with_strength::<A>(Arc::clone(&ctx), strength)?,
            tree_builder_state: None,
            max_batch_size,
            _a: PhantomData::<A>,
        };

        let ptr = {
            let ctx = ctx.lock().unwrap();
            FutharkContextPointer(&*ctx)
        };

        let mut locked = BATCH_HASHERS.lock().unwrap();
        let entry = (*locked).entry(ptr).or_default();
        *entry += 1;

        Ok(new)
    }

    pub(crate) fn futhark_context(&self) -> Arc<Mutex<FutharkContext>> {
        self.ctx.clone()
    }
}

impl<A> Drop for GpuBatchHasher<A> {
    fn drop(&mut self) {
        let ctx = self.ctx.lock().unwrap();
        let mut locked = BATCH_HASHERS.lock().unwrap();
        let entry = (*locked).entry(FutharkContextPointer(&*ctx)).or_default();

        *entry -= 1;

        if *entry == 0 {
            // SAFETY: this is the last instance associated with the context,
            // and a new one cannot be created while we hold the Mutex.
            // So there will be no concurrent uses of the context while we clear caches.
            unsafe {
                triton::bindings::futhark_context_clear_caches(ctx.context);
            }
        };
    }
}

impl<A> BatchHasher<A> for GpuBatchHasher<A>
where
    A: Arity<Fr>,
{
    /// Hash a batch of `A`-sized preimages.
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        let mut ctx = self.ctx.lock().unwrap();
        let (res, state) = self.state.hash(&mut ctx, preimages)?;
        self.state = state;
        Ok(res)
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[derive(Debug)]
struct GpuConstants<A>(PoseidonConstants<Fr, A>)
where
    A: Arity<Fr>;

impl<A> GpuConstants<A>
where
    A: Arity<Fr>,
{
    fn arity_tag(&self, ctx: &FutharkContext) -> Result<Array_u64_1d, Error> {
        let arity_tag = self.0.domain_tag;
        array_u64_1d_from_fr(ctx, arity_tag)
    }

    fn round_keys(&self, ctx: &FutharkContext) -> Result<Array_u64_2d, Error> {
        let round_keys = &self.0.compressed_round_constants;
        array_u64_2d_from_frs(ctx, &round_keys)
    }

    fn mds_matrix(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let matrix = &self.0.mds_matrices.m;

        array_u64_3d_from_frs_2d(ctx, matrix)
    }

    fn pre_sparse_matrix(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let pre_sparse_matrix = &self.0.pre_sparse_matrix;

        array_u64_3d_from_frs_2d(ctx, pre_sparse_matrix)
    }

    fn sparse_matrixes(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let sparse_matrixes = &self.0.sparse_matrixes;

        let frs_2d: Vec<Vec<Fr>> = sparse_matrixes
            .iter()
            .map(|m| {
                let mut x = m.w_hat.clone();
                x.extend(m.v_rest.clone());
                x.into_iter().collect()
            })
            .collect();

        array_u64_3d_from_frs_2d(ctx, &frs_2d)
    }
}

/// Conversion functions for massaging input/output to and from interface types.
fn frs_to_u64s(frs: &[Fr]) -> Vec<u64> {
    let mut res = vec![u64::default(); frs.len() * 4];
    for (src, dest) in frs.iter().zip(res.chunks_mut(4)) {
        dest.copy_from_slice(&src.into_repr().0);
    }
    res
}

fn frs_2d_to_u64s(frs_2d: &[Vec<Fr>]) -> Vec<u64> {
    frs_2d
        .iter()
        .flat_map(|row| frs_to_u64s(row).into_iter())
        .collect()
}

fn array_u64_1d_from_fr(ctx: &FutharkContext, fr: Fr) -> Result<Array_u64_1d, Error> {
    Array_u64_1d::from_vec(*ctx, &fr.into_repr().0, &[4, 1])
        .map_err(|e| Error::Other(format!("error converting Fr: {:?}", e)))
}

fn array_u64_1d_from_frs(ctx: &FutharkContext, frs: &[Fr]) -> Result<Array_u64_1d, Error> {
    let u64s = frs_to_u64s(frs);

    Array_u64_1d::from_vec(*ctx, u64s.as_slice(), &[(frs.len() * 4) as i64, 1])
        .map_err(|e| Error::Other(format!("error converting Fr: {:?}", e)))
}

fn array_u64_2d_from_frs(ctx: &FutharkContext, frs: &[Fr]) -> Result<Array_u64_2d, Error> {
    let u64s = frs_to_u64s(frs);

    let d2 = 4;
    let d1 = u64s.len() as i64 / d2;
    let dim = [d1, d2];

    Array_u64_2d::from_vec(*ctx, u64s.as_slice(), &dim)
        .map_err(|e| Error::Other(format!("error converting Frs: {:?}", e)))
}

fn array_u64_3d_from_frs_2d(
    ctx: &FutharkContext,
    frs_2d: &[Vec<Fr>],
) -> Result<Array_u64_3d, Error> {
    let u64s = frs_2d_to_u64s(frs_2d);

    let mut l = u64s.len() as i64;
    let d1 = 4; // One Fr is 4 x u64.
    l /= d1;

    let d2 = frs_2d[0].len() as i64;
    assert!(
        frs_2d.iter().all(|x| x.len() == d2 as usize),
        "Frs must be grouped uniformly"
    );
    l /= d2;

    let d3 = l as i64;
    let dim = [d3, d2, d1];

    Array_u64_3d::from_vec(*ctx, u64s.as_slice(), &dim)
        .map_err(|e| Error::Other(format!("error converting Frs 2d: {:?}", e)))
}

pub fn u64s_into_fr(limbs: &[u64]) -> Result<Fr, PrimeFieldDecodingError> {
    assert_eq!(limbs.len(), 4);
    let limb_arr = limbs.try_into().expect("Correct length was asserted.");
    let repr = FrRepr(limb_arr);
    Fr::from_repr(repr)
}

fn unpack_fr_array(vec_shape: (Vec<u64>, &[i64])) -> Result<Vec<Fr>, Error> {
    let (vec, shape) = vec_shape;
    let chunk_size = shape[shape.len() - 1] as usize;

    vec.chunks(chunk_size)
        .map(|x| u64s_into_fr(x))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::DecodingError)
}

fn unpack_fr_array_from_monts(monts: &[u64]) -> Result<&[Fr], Error> {
    let fr_size = 4;
    let fr_count = monts.len() / fr_size;
    assert_eq!(
        std::mem::size_of::<Fr>(),
        fr_size * std::mem::size_of::<u64>()
    );

    if monts.len() % fr_size != 0 {
        return Err(Error::Other(
            "wrong size monts to convert to Frs".to_string(),
        ));
    }

    let frs = unsafe { std::slice::from_raw_parts(monts.as_ptr() as *const () as _, fr_count) };

    Ok(frs)
}

fn as_mont_u64s<U: ArrayLength<Fr>>(vec: &[GenericArray<Fr, U>]) -> &[u64] {
    let fr_size = 4; // Number of limbs in Fr.
    assert_eq!(
        fr_size * std::mem::size_of::<u64>(),
        std::mem::size_of::<Fr>(),
        "fr size changed"
    );

    unsafe {
        std::slice::from_raw_parts(
            vec.as_ptr() as *const () as *const u64,
            vec.len() * fr_size * U::to_usize(),
        )
    }
}

fn frs_as_mont_u64s(vec: &[Fr]) -> &[u64] {
    let fr_size = 4; // Number of limbs in Fr.
    assert_eq!(
        fr_size * std::mem::size_of::<u64>(),
        std::mem::size_of::<Fr>(),
        "fr size changed"
    );

    unsafe {
        std::slice::from_raw_parts(vec.as_ptr() as *const () as *const u64, vec.len() * fr_size)
    }
}

fn as_u64s<U: ArrayLength<Fr>>(vec: &[GenericArray<Fr, U>]) -> Vec<u64> {
    if vec.is_empty() {
        return Vec::new();
    }
    let fr_size = std::mem::size_of::<Fr>();
    let mut safely = Vec::with_capacity(vec.len() * U::to_usize() * fr_size);
    for row in vec {
        for column in row {
            safely.extend_from_slice(&column.into_repr().0)
        }
    }
    safely
}

fn init_hash2(ctx: &mut FutharkContext, strength: Strength) -> Result<BatcherState, Error> {
    let constants = GpuConstants(PoseidonConstants::<Fr, U2>::new_with_strength(strength));
    match strength {
        Strength::Standard => {
            let state = ctx
                .init2(
                    constants.arity_tag(&ctx)?,
                    constants.round_keys(&ctx)?,
                    constants.mds_matrix(&ctx)?,
                    constants.pre_sparse_matrix(&ctx)?,
                    constants.sparse_matrixes(&ctx)?,
                )
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
            Ok(BatcherState::Arity2(state))
        }
        Strength::Strengthened => {
            let state = ctx
                .init2s(
                    constants.arity_tag(&ctx)?,
                    constants.round_keys(&ctx)?,
                    constants.mds_matrix(&ctx)?,
                    constants.pre_sparse_matrix(&ctx)?,
                    constants.sparse_matrixes(&ctx)?,
                )
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
            Ok(BatcherState::Arity2s(state))
        }
    }
}

fn init_hash8(ctx: &mut FutharkContext, strength: Strength) -> Result<BatcherState, Error> {
    let constants = GpuConstants(PoseidonConstants::<Fr, U8>::new_with_strength(strength));
    match strength {
        Strength::Standard => {
            let state = ctx
                .init8(
                    constants.arity_tag(&ctx)?,
                    constants.round_keys(&ctx)?,
                    constants.mds_matrix(&ctx)?,
                    constants.pre_sparse_matrix(&ctx)?,
                    constants.sparse_matrixes(&ctx)?,
                )
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

            Ok(BatcherState::Arity8(state))
        }
        Strength::Strengthened => {
            let state = ctx
                .init8s(
                    constants.arity_tag(&ctx)?,
                    constants.round_keys(&ctx)?,
                    constants.mds_matrix(&ctx)?,
                    constants.pre_sparse_matrix(&ctx)?,
                    constants.sparse_matrixes(&ctx)?,
                )
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

            Ok(BatcherState::Arity8s(state))
        }
    }
}

fn init_hash11(ctx: &mut FutharkContext, strength: Strength) -> Result<BatcherState, Error> {
    let constants = GpuConstants(PoseidonConstants::<Fr, U11>::new_with_strength(strength));

    match strength {
        Strength::Standard => {
            let state = ctx
                .init11(
                    constants.arity_tag(&ctx)?,
                    constants.round_keys(&ctx)?,
                    constants.mds_matrix(&ctx)?,
                    constants.pre_sparse_matrix(&ctx)?,
                    constants.sparse_matrixes(&ctx)?,
                )
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

            Ok(BatcherState::Arity11(state))
        }
        Strength::Strengthened => {
            let state = ctx
                .init11s(
                    constants.arity_tag(&ctx)?,
                    constants.round_keys(&ctx)?,
                    constants.mds_matrix(&ctx)?,
                    constants.pre_sparse_matrix(&ctx)?,
                    constants.sparse_matrixes(&ctx)?,
                )
                .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

            Ok(BatcherState::Arity11s(state))
        }
    }
}

fn mbatch_hash2<A>(
    ctx: &mut FutharkContext,
    state: &mut P2State,
    preimages: &[GenericArray<Fr, A>],
) -> Result<(Vec<Fr>, P2State), Error>
where
    A: Arity<Fr>,
{
    assert_eq!(2, A::to_usize());
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash2(state, input)
        .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec()?;
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn mbatch_hash8<A>(
    ctx: &mut FutharkContext,
    state: &P8State,
    preimages: &[GenericArray<Fr, A>],
) -> Result<(Vec<Fr>, P8State), Error>
where
    A: Arity<Fr>,
{
    assert_eq!(8, A::to_usize());
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash8(state, input)
        .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec()?;
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn mbatch_hash11<A>(
    ctx: &mut FutharkContext,
    state: &P11State,
    preimages: &[GenericArray<Fr, A>],
) -> Result<(Vec<Fr>, P11State), Error>
where
    A: Arity<Fr>,
{
    assert_eq!(11, A::to_usize());
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash11(state, input)
        .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec()?;
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn mbatch_hash2s<A>(
    ctx: &mut FutharkContext,
    state: &mut S2State,
    preimages: &[GenericArray<Fr, A>],
) -> Result<(Vec<Fr>, S2State), Error>
where
    A: Arity<Fr>,
{
    assert_eq!(2, A::to_usize());
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash2s(state, input)
        .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec()?;
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn mbatch_hash8s<A>(
    ctx: &mut FutharkContext,
    state: &S8State,
    preimages: &[GenericArray<Fr, A>],
) -> Result<(Vec<Fr>, S8State), Error>
where
    A: Arity<Fr>,
{
    assert_eq!(8, A::to_usize());
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash8s(state, input)
        .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec()?;
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn mbatch_hash11s<A>(
    ctx: &mut FutharkContext,
    state: &S11State,
    preimages: &[GenericArray<Fr, A>],
) -> Result<(Vec<Fr>, S11State), Error>
where
    A: Arity<Fr>,
{
    assert_eq!(11, A::to_usize());
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash11s(state, input)
        .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec()?;
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn u64_vec<U: ArrayLength<Fr>>(vec: &[GenericArray<Fr, U>]) -> Vec<u64> {
    vec![0; vec.len() * U::to_usize() * std::mem::size_of::<Fr>()]
}

#[cfg(test)]
/// GPU performance is still inexplicably poor and behavior erratic on macos,
/// So skip GPU tests by default since real code paths are now enabled.
/// Users will probably not want to actually run with GPU on macos,
/// But if experiments show it is viable, then it is possible.
#[cfg(all(feature = "futhark", not(target_os = "macos")))]
mod tests {
    use super::*;
    use crate::poseidon::{Poseidon, SimplePoseidonBatchHasher};
    use crate::triton::gpu::BatcherState;
    use crate::BatchHasher;
    use fff::{Field, ScalarEngine};
    use generic_array::sequence::GenericSequence;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_mbatch_hash2() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let mut state = if let BatcherState::Arity2(s) =
            init_hash2(&mut ctx.lock().unwrap(), Strength::Standard).unwrap()
        {
            s
        } else {
            panic!("expected Arity2");
        };
        let batch_size = 100;

        let mut gpu_hasher = GpuBatchHasher::<U2>::new_with_strength(
            cl::default_futhark_context().unwrap(),
            Strength::Standard,
            batch_size,
        )
        .unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U2>::new_with_strength(Strength::Standard, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash2(&mut ctx.lock().unwrap(), &mut state, preimages.as_slice()).unwrap();

        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }

    #[test]
    fn test_mbatch_hash2s() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let mut state = if let BatcherState::Arity2s(s) =
            init_hash2(&mut ctx.lock().unwrap(), Strength::Strengthened).unwrap()
        {
            s
        } else {
            panic!("expected Arity2s");
        };
        let batch_size = 100;

        let mut gpu_hasher = GpuBatchHasher::<U2>::new_with_strength(
            cl::default_futhark_context().unwrap(),
            Strength::Strengthened,
            batch_size,
        )
        .unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U2>::new_with_strength(Strength::Strengthened, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash2s(&mut ctx.lock().unwrap(), &mut state, preimages.as_slice()).unwrap();
        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }

    #[test]
    fn test_mbatch_hash8() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let state = if let BatcherState::Arity8(s) =
            init_hash8(&mut ctx.lock().unwrap(), Strength::Standard).unwrap()
        {
            s
        } else {
            panic!("expected Arity8");
        };
        let batch_size = 100;

        let mut gpu_hasher = GpuBatchHasher::<U8>::new_with_strength(
            cl::default_futhark_context().unwrap(),
            Strength::Standard,
            batch_size,
        )
        .unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U8>::new_with_strength(Strength::Standard, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U8>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash8(&mut ctx.lock().unwrap(), &state, preimages.as_slice()).unwrap();
        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }

    #[test]
    fn test_mbatch_hash8s() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let state = if let BatcherState::Arity8s(s) =
            init_hash8(&mut ctx.lock().unwrap(), Strength::Strengthened).unwrap()
        {
            s
        } else {
            panic!("expected Arity8s");
        };
        let batch_size = 100;

        let mut gpu_hasher = GpuBatchHasher::<U8>::new_with_strength(
            cl::default_futhark_context().unwrap(),
            Strength::Strengthened,
            batch_size,
        )
        .unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U8>::new_with_strength(Strength::Strengthened, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U8>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash8s(&mut ctx.lock().unwrap(), &state, preimages.as_slice()).unwrap();
        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }

    #[test]
    fn test_mbatch_hash11() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let state = if let BatcherState::Arity11(s) =
            init_hash11(&mut ctx.lock().unwrap(), Strength::Standard).unwrap()
        {
            s
        } else {
            panic!("expected Arity11");
        };
        let batch_size = 100;

        let mut gpu_hasher = GpuBatchHasher::<U11>::new_with_strength(
            cl::default_futhark_context().unwrap(),
            Strength::Standard,
            batch_size,
        )
        .unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U11>::new_with_strength(Strength::Standard, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U11>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash11(&mut ctx.lock().unwrap(), &state, preimages.as_slice()).unwrap();
        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }

    #[test]
    fn test_mbatch_hash11s() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let state = if let BatcherState::Arity11s(s) =
            init_hash11(&mut ctx.lock().unwrap(), Strength::Strengthened).unwrap()
        {
            s
        } else {
            panic!("expected Arity11s");
        };
        let batch_size = 100;

        let mut gpu_hasher = GpuBatchHasher::<U11>::new_with_strength(
            cl::default_futhark_context().unwrap(),
            Strength::Strengthened,
            batch_size,
        )
        .unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U11>::new_with_strength(Strength::Strengthened, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U11>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash11s(&mut ctx.lock().unwrap(), &state, preimages.as_slice()).unwrap();
        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }

    fn test_mbatch_hash8_on_device(dev: Arc<Mutex<FutharkContext>>) {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let ctx = cl::default_futhark_context().unwrap();
        let state = if let BatcherState::Arity8(s) =
            init_hash8(&mut ctx.lock().unwrap(), Strength::Standard).unwrap()
        {
            s
        } else {
            panic!("expected Arity8");
        };
        let batch_size = 100;

        let mut gpu_hasher =
            GpuBatchHasher::<U8>::new_with_strength(dev, Strength::Standard, batch_size).unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U8>::new_with_strength(Strength::Standard, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U8>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, _) =
            mbatch_hash8(&mut ctx.lock().unwrap(), &state, preimages.as_slice()).unwrap();
        let gpu_hashes = gpu_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        assert_eq!(expected_hashes, hashes);
        assert_eq!(expected_hashes, gpu_hashes);
    }
}
