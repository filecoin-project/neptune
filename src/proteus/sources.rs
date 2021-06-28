use super::gpu::DerivedConstants;
use bellperson::bls::{Engine, Fr};
use fff::PrimeField;
use itertools::join;

fn config() -> String {
    "".to_string()
}

fn poseidon_source(field: &str, derived_constants: DerivedConstants) -> String {
    let DerivedConstants {
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
    } = derived_constants;

    format!(
        include_str!("cl/poseidon.cl"),
        arity = arity,
        field = field,
        partial_rounds = partial_rounds,
        width = width,
        sparse_matrix_size = sparse_matrix_size,
        full_half = full_half,
        sparse_offset = sparse_offset,
        constants_elements = constants_elements,
        domain_tag_offset = domain_tag_offset,
        round_keys_offset = round_keys_offset,
        mds_matrix_offset = mds_matrix_offset,
        pre_sparse_matrix_offset = pre_sparse_matrix_offset,
        w_hat_offset = w_hat_offset,
        v_rest_offset = v_rest_offset,
        sparse_matrixes_offset = sparse_matrixes_offset,
    )
}

pub fn generate_program<Fr>(limb64: bool, derived_constants: DerivedConstants) -> String
where
    Fr: PrimeField,
{
    let field_source = if limb64 {
        ff_cl_gen::field::<Fr, ff_cl_gen::Limb64>("Fr")
    } else {
        ff_cl_gen::field::<Fr, ff_cl_gen::Limb32>("Fr")
    };
    join(
        &[
            config(),
            field_source,
            poseidon_source("Fr", derived_constants),
        ],
        "\n",
    )
}
