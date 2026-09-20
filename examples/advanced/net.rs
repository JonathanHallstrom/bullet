use bullet_lib::game::{
    inputs::{ChessBucketsMirrored, SparseInputType},
    outputs::MaterialCount,
};
use bullet_trainer::model::{InitSettings, ModelDefinition, ModelInputs};

use crate::inputs::{self, InputTy, PawnPawnInputs};

pub const L1: usize = 768;
pub const L2: usize = 16;
pub const L3: usize = 32;
pub const OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
pub const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

pub type Network =
    (ModelInputs<InputTy>, PawnPawnInputs, ChessBucketsMirrored, MaterialCount<OUTPUT_BUCKETS>, ModelDefinition);

pub fn build() -> Network {
    let pp = PawnPawnInputs::new(inputs::three_file_band_mask());
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let output_buckets = MaterialCount::<OUTPUT_BUCKETS>;

    let inputs = ModelInputs::default()
        .add_sparse("stm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("ntm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("buckets", (OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), output_buckets), target)| {
            let l0_pp = builder.new_affine("l0/pp/", pp.num_inputs(), L1);

            let l0f = builder.new_weights("l0/fac", (L1, 768), InitSettings::Zeroed);
            let psqt_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
            let mut l0_psqt = builder.new_weights("l0/psqt", (L1, psqt.num_inputs()), psqt_init);
            l0_psqt = l0_psqt + l0f.repeat(psqt.num_inputs() / 768);

            let l1 = builder.new_affine("l1/", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2/", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3/", L3, OUTPUT_BUCKETS);

            let ft = |pp, psqt, start, end| {
                (l0_pp.slice(start, end).forward(pp) + l0_psqt.slice_rows(start, end).matmul(psqt)).crelu()
            };
            let stm_hidden = ft(stm_pp, stm_psqt, 0, L1 / 2) * ft(stm_pp, stm_psqt, L1 / 2, L1);
            let ntm_hidden = ft(ntm_pp, ntm_psqt, 0, L1 / 2) * ft(ntm_pp, ntm_psqt, L1 / 2, L1);
            let l0_out = stm_hidden.concat(ntm_hidden);
            let l0_out_norm = l0_out.reduce_sum_rows() / (L1 as f32);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();
            let l2_out = l2.forward(hl2).select(output_buckets);
            let l3_out = l3.forward(l2_out.crelu()).select(output_buckets);
            let loss = l3_out.sigmoid().squared_error(target) + 0.005 * l0_out_norm;

            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), l3_out)])
        },
    );

    (inputs, pp, psqt, output_buckets, defn)
}
