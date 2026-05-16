/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
*/
use bullet_lib::{
    game::{
        inputs::{self, SparseInputType, get_num_buckets},
        outputs::{self, OutputBuckets},
    },
    nn::{
        InitSettings, Shape,
        optimiser::{self, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::ViriBinpackLoader},
};
use bulletformat::ChessBoard;

const L1: usize = 1024;
const L2: usize = 16;
const L3: usize = 32;
const SCALE: i32 = 400;
const SUPERBATCHES: usize = 400;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    // 0,  1,  2,  3,
    // 4,  5,  6,  7,
    // 8,  8,  8,  8,
    // 9,  9,  9,  9,
    // 10, 10, 10, 10,
    // 10, 10, 10, 10,
    // 11, 11, 11, 11,
    // 11, 11, 11, 11,
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
];
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const MATERIAL_BUCKETS: usize = 8;
const OUTPUT_BUCKETS: usize = MATERIAL_BUCKETS * INPUT_BUCKETS;

#[derive(Clone, Copy, Default)]
pub struct KingMaterialCount<const M: usize, const K: usize> {
    pub king: inputs::ChessBucketsMirrored,
}

impl<const M: usize, const K: usize> KingMaterialCount<M, K> {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self { king: inputs::ChessBucketsMirrored::new(buckets) }
    }
}

impl<const M: usize, const K: usize> outputs::OutputBuckets<bulletformat::ChessBoard> for KingMaterialCount<M, K> {
    const BUCKETS: usize = M * K;

    fn bucket(&self, pos: &bulletformat::ChessBoard) -> u8 {
        let divisor = 62_u8.div_ceil(M as u8);
        let material_bucket = {
            let mut res: u8 = 0;

            #[rustfmt::skip]
            const WEIGHTS: [u8; 16] = [
                0, 3, 3, 5, 9, 0, 0, 0,
                0, 3, 3, 5, 9, 0, 0, 0,
            ];

            let num_pieces = pos.occ().count_ones() as usize;
            for &pc in &pos.pcs[0..num_pieces.div_ceil(2)] {
                res += WEIGHTS[(pc & 0xf) as usize];
                res += WEIGHTS[(pc >> 4) as usize];
            }

            res / divisor
        };

        let mut king_bucket = 0;
        let inputs_per_bucket = self.king.num_inputs() / K;
        self.king.map_features(pos, |stm, _| {
            king_bucket = (stm / inputs_per_bucket) as u8;
        });

        king_bucket * M as u8 + material_bucket
    }
}

impl<const M: usize, const K: usize> outputs::OutputBuckets<bulletformat::chess::MarlinFormat>
    for KingMaterialCount<M, K>
{
    const BUCKETS: usize = M * K;

    fn bucket(&self, pos: &bulletformat::chess::MarlinFormat) -> u8 {
        return self.bucket(&bulletformat::ChessBoard::from(*pos));
    }
}

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

fn main() {
    {
        let pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0".parse::<ChessBoard>().unwrap();
        let buckets = KingMaterialCount::<MATERIAL_BUCKETS, INPUT_BUCKETS>::new(BUCKET_LAYOUT);
        println!("{}", buckets.bucket(&pos));
    }

    let mut trainer = ValueTrainerBuilder::default()
        // makes `ntm_inputs` available below
        .dual_perspective()
        // standard optimiser used in NNUE
        // the default AdamW params include clipping to range [-1.98, 1.98]
        .optimiser(optimiser::AdamW)
        // basic piece-square chessboard inputs
        .inputs(inputs::ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(KingMaterialCount::<MATERIAL_BUCKETS, INPUT_BUCKETS>::new(BUCKET_LAYOUT))
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|builder, mut weights| {
                    let expanded = builder.get("l0f").values.f32().repeat(INPUT_BUCKETS);

                    for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                        *i += j;
                    }

                    weights
                })
                .round()
                .quantise::<i16>(Q0),
            SavedFormat::id("l0b").round().quantise::<i16>(Q0),
            SavedFormat::id("l1w")
                .transform(|_, mut weights| {
                    for i in weights.iter_mut() {
                        *i /= FT_SHIFT_SCALE * FT_SHIFT_SCALE;
                    }
                    weights
                })
                .round()
                .quantise::<i8>(Q1),
            SavedFormat::id("l1b").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2b").round().quantise::<i32>((Q as i32).pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l3b").round().quantise::<i32>((Q as i32).pow(4)),
        ])
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), target| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, L1);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);

            // let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            // let l0_out_norm = ones_l1_vec.matmul(hl1);

            let l1_out = l1.forward(hl1).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().squared_error(target);

            // let loss = loss + 0.005 * l0_out_norm;

            (l3_out, loss)
        });
    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0f", l0_clip);

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let schedule = TrainingSchedule {
        net_id: "kirill_1024_pw_8ib_lin_crazy_buckets".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384 * 8,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.3, end: 0.7 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 1e-3,
            final_lr: 1e-3 * 0.3f32.powi(5),
            final_superbatch: SUPERBATCHES,
        },

        save_rate: 100,
    };

    let settings = LocalSettings { threads: 16, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    // loading from a SF binpack

    let data_loader = {
        let file_path = "/k4/kirill_data/2026_04_25/combined.vf";
        let buffer_size_mb = 8192;
        let threads = 16;
        ViriBinpackLoader::new(file_path, buffer_size_mb, threads, viriformat::dataformat::Filter::default())
        // fn filter(entry: &TrainingDataEntry) -> bool {
        //     !entry.pos.is_checked(entry.pos.side_to_move())
        //         && entry.score.unsigned_abs() <= 10000
        //         && entry.mv.mtype() == MoveType::Normal
        //         && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        // }
    };

    // loading directly from a `BulletFormat` file
    // let data_loader = loader::DirectSequentialDataLoader::new(&["/k4/kirill_data/2026_01_01/pos_sanitised.bf"]);

    // trainer.load_from_checkpoint("checkpoints/kirill_1024_pw_ib-200");
    // trainer.save_to_checkpoint("checkpoints/kirill_1024_pw_ib-200");
    trainer.run(&schedule, &settings, &data_loader);
    // trainer.load_from_checkpoint("checkpoints/kirill_512_multi-200");

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
