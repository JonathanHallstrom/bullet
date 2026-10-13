use bullet_lib::{
    game::inputs::{ChessBucketsMirrored, SparseInputType},
    nn::InitSettings,
};
use bullet_trainer::model::{ModelDefinition, ModelInputs};

use crate::inputs::{self, InputTy, PawnPawnInputs};

pub const L1_SIZE: usize = 4096;
pub const L2_SIZE: usize = 128;
pub const L3_SIZE: usize = 256;

#[rustfmt::skip]
pub const BUCKET_LAYOUT: [usize; 32] = [
     0,  0,  1,  1,
     2,  2,  2,  2,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
];

pub type Network = (ModelInputs<InputTy>, PawnPawnInputs, ChessBucketsMirrored, ModelDefinition);

pub fn build() -> Network {
    let pp = PawnPawnInputs::new(inputs::three_file_band_mask());
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);

    let inputs = ModelInputs::default()
        .add_sparse("stm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("ntm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(&inputs, |builder, ((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), target)| {
        let l0_pp = builder.new_affine("l0/pp/", pp.num_inputs(), L1_SIZE);

        let l0f = builder.new_weights("l0/fac", (L1_SIZE, 768), InitSettings::Zeroed);

        let psqt_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
        let mut l0_psqt = builder.new_weights("l0/psqt", (L1_SIZE, psqt.num_inputs()), psqt_init);
        l0_psqt = l0_psqt + l0f.repeat(psqt.num_inputs() / 768);

        let l1 = builder.new_affine("l1/", L1_SIZE, L2_SIZE);
        let l2 = builder.new_affine("l2/", L2_SIZE * 2, L3_SIZE);
        let l3 = builder.new_affine("l3/", L3_SIZE, 1);

        let ft = |pp_in, psqt_in, start, end| {
            (l0_pp.slice(start, end).forward(pp_in) + l0_psqt.slice_rows(start, end).matmul(psqt_in)).crelu()
        };
        let stm_hidden = ft(stm_pp, stm_psqt, 0, L1_SIZE / 2) * ft(stm_pp, stm_psqt, L1_SIZE / 2, L1_SIZE);
        let ntm_hidden = ft(ntm_pp, ntm_psqt, 0, L1_SIZE / 2) * ft(ntm_pp, ntm_psqt, L1_SIZE / 2, L1_SIZE);
        let hl1 = stm_hidden.concat(ntm_hidden);

        let l0_out_norm = hl1.reduce_sum_rows() / (L1_SIZE as f32);

        let l1_out = l1.forward(hl1);
        let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

        let hl3 = l2.forward(hl2).screlu();
        let out = l3.forward(hl3);

        let loss = out.sigmoid().squared_error(target);
        let loss = loss + 0.005 * l0_out_norm;

        (Some(loss.reduce_sum_batch()), vec![("output".to_string(), out)])
    });

    (inputs, pp, psqt, defn)
}
