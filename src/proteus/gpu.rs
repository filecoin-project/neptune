use super::sources::generate_program;
use crate::error::{ClError, Error};
use crate::hash_type::HashType;
use crate::poseidon::PoseidonConstants;
use crate::{Arity, BatchHasher, Strength, DEFAULT_STRENGTH};
use bellperson::bls::{Bls12, Fr, FrRepr};
use ff::{Field, PrimeField, PrimeFieldDecodingError};
use generic_array::{typenum, ArrayLength, GenericArray};
use log::info;
use rust_gpu_tools::opencl::{self, cl_device_id, Device};
use std::collections::HashMap;
use std::marker::PhantomData;
use typenum::{U11, U2, U8};

#[derive(Debug)]
struct GpuConstants<A>(PoseidonConstants<Bls12, A>)
where
    A: Arity<Fr>;

pub struct ClBatchHasher<A>
where
    A: Arity<Fr>,
{
    device: opencl::Device,
    constants: GpuConstants<A>,
    constants_buffer: opencl::Buffer<Fr>,
    max_batch_size: usize,
    program: opencl::Program,
}

pub struct DerivedConstants {
    pub arity: usize,
    pub partial_rounds: usize,
    pub width: usize,
    pub sparse_matrix_size: usize,
    pub full_half: usize,
    pub sparse_offset: usize,
    pub constants_elements: usize,

    // Offsets
    pub domain_tag_offset: usize,
    pub round_keys_offset: usize,
    pub mds_matrix_offset: usize,
    pub pre_sparse_matrix_offset: usize,
    pub sparse_matrixes_offset: usize,
    pub w_hat_offset: usize,
    pub v_rest_offset: usize,
}

impl<A> GpuConstants<A>
where
    A: Arity<Fr>,
{
    #[allow(clippy::suspicious_operation_groupings)]
    fn derived_constants(&self) -> DerivedConstants {
        let c = &self.0;
        let arity = c.arity();
        let full_rounds = c.full_rounds;
        let partial_rounds = c.partial_rounds;
        let sparse_count = partial_rounds;
        let width = arity + 1;
        let sparse_matrix_size = 2 * width - 1;
        let rk_count = width * full_rounds + partial_rounds;
        let full_half = full_rounds / 2;
        let sparse_offset = full_half - 1;
        let constants_elements =
            1 + rk_count + (width * width) + (width * width) + (sparse_count * sparse_matrix_size);

        let matrix_size = width * width;
        let mut offset = 0;
        let domain_tag_offset = offset;
        offset += 1;
        let round_keys_offset = offset;
        offset += rk_count;
        let mds_matrix_offset = offset;
        offset += matrix_size;
        let pre_sparse_matrix_offset = offset;
        offset += matrix_size;
        let sparse_matrixes_offset = offset;

        let w_hat_offset = 0;
        let v_rest_offset = width;

        DerivedConstants {
            arity,
            partial_rounds,
            width,
            sparse_matrix_size,
            full_half,
            sparse_offset,
            constants_elements,
            domain_tag_offset,
            round_keys_offset,
            mds_matrix_offset,
            pre_sparse_matrix_offset,
            sparse_matrixes_offset,
            w_hat_offset,
            v_rest_offset,
        }
    }
}

impl<A> GpuConstants<A>
where
    A: Arity<Fr>,
{
    fn full_rounds(&self) -> usize {
        self.0.full_rounds
    }

    fn partial_rounds(&self) -> usize {
        self.0.partial_rounds
    }

    fn to_buffer(&self, program: &opencl::Program) -> Result<opencl::Buffer<Fr>, Error> {
        let DerivedConstants {
            arity: _,
            partial_rounds: _,
            width: _,
            sparse_matrix_size: _,
            full_half: _,
            sparse_offset: _,
            constants_elements,
            domain_tag_offset,
            round_keys_offset,
            mds_matrix_offset,
            pre_sparse_matrix_offset,
            sparse_matrixes_offset,
            w_hat_offset: _,
            v_rest_offset: _,
        } = self.derived_constants();

        let buffer = program
            .create_buffer::<Fr>(constants_elements)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

        let c = &self.0;

        program
            .write_from_buffer(&buffer, domain_tag_offset, &[c.domain_tag])
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        program
            .write_from_buffer(&buffer, round_keys_offset, &c.compressed_round_constants)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        program
            .write_from_buffer(
                &buffer,
                mds_matrix_offset,
                c.mds_matrices
                    .m
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        program
            .write_from_buffer(
                &buffer,
                pre_sparse_matrix_offset,
                c.pre_sparse_matrix
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        let mut sm_elts = Vec::new();
        for sm in c.sparse_matrixes.iter() {
            sm_elts.extend(sm.w_hat.iter());
            sm_elts.extend(sm.v_rest.iter());
        }
        program
            .write_from_buffer(&buffer, sparse_matrixes_offset, &sm_elts)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

        Ok(buffer)
    }
}

impl<A> ClBatchHasher<A>
where
    A: Arity<Fr>,
{
    /// Create a new `GPUBatchHasher` and initialize it with state corresponding with its `A`.
    pub(crate) fn new(device: &opencl::Device, max_batch_size: usize) -> Result<Self, Error> {
        Self::new_with_strength(device, DEFAULT_STRENGTH, max_batch_size)
    }

    pub(crate) fn new_with_strength(
        device: &opencl::Device,
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        let constants = GpuConstants(PoseidonConstants::<Bls12, A>::new_with_strength(strength));
        let src = generate_program::<Fr>(true, constants.derived_constants());
        let program = opencl::Program::from_opencl(&device, &src)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        let constants_buffer = constants.to_buffer(&program)?;
        Ok(Self {
            device: device.clone(),
            constants,
            constants_buffer,
            max_batch_size,
            program,
        })
    }

    pub(crate) fn device(&self) -> opencl::Device {
        self.device.clone()
    }
}
const LOCAL_WORK_SIZE: usize = 256;
impl<A> BatchHasher<A> for ClBatchHasher<A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        let local_work_size = LOCAL_WORK_SIZE;
        let max_batch_size = self.max_batch_size;
        let batch_size = preimages.len();
        assert!(batch_size <= max_batch_size);

        // Set `global_work_size` to smallest multiple of `local_work_size` >= `batch-size`.
        let global_work_size = ((batch_size / local_work_size)
            + (batch_size % local_work_size != 0) as usize)
            * local_work_size;

        let num_hashes = preimages.len();

        let kernel = self
            .program
            .create_kernel("hash_preimages", global_work_size, local_work_size)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

        let preimages_buffer = self
            .program
            .create_buffer::<GenericArray<Fr, A>>(num_hashes)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

        self.program
            .write_from_buffer(&preimages_buffer, 0, preimages)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        let result_buffer = self
            .program
            .create_buffer::<Fr>(num_hashes)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

        kernel
            .arg(&self.constants_buffer)
            .arg(&preimages_buffer)
            .arg(&result_buffer)
            .arg(&(preimages.len() as i32))
            .run()
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;

        let mut frs = vec![<Fr as Field>::zero(); num_hashes];
        self.program
            .read_into_buffer(&result_buffer, 0, &mut frs)
            .map_err(|e| Error::GpuError(format!("{:?}", e)))?;
        Ok(frs.to_vec())
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[cfg(test)]
#[cfg(all(feature = "opencl", not(target_os = "macos")))]
mod test {
    use super::*;
    use crate::poseidon::{Poseidon, SimplePoseidonBatchHasher};
    use generic_array::sequence::GenericSequence;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_batch_hash_2() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let all = opencl::Device::all();
        let device = all.first().unwrap();

        // NOTE: `batch_size` is not a multiple of `LOCAL_WORK_SIZE`.
        let batch_size = 1025;

        let mut cl_hasher =
            ClBatchHasher::<U2>::new_with_strength(device, Strength::Standard, batch_size).unwrap();
        let mut simple_hasher =
            SimplePoseidonBatchHasher::<U2>::new_with_strength(Strength::Standard, batch_size);

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let cl_hashes = cl_hasher.hash(&preimages).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages).unwrap();

        dbg!(
            &cl_hashes,
            &expected_hashes,
            &cl_hashes.len(),
            &expected_hashes.len()
        );

        assert_eq!(expected_hashes, cl_hashes);
    }
}
