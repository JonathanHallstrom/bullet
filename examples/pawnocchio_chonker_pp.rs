use std::sync::atomic::{AtomicU64, Ordering};

use bullet_lib::{
    game::{inputs::SparseInputType, outputs::MaterialCount},
    nn::{
        Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{ViriBinpackLoader, viribinpack::ViriFilter},
    },
};
use bytemuck::zeroed;
use rand::{Rng, rng};
use viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::{Filter, WDL},
};

type Optimiser = AdamW;
type OptimiserParams = AdamWParams;
const NET_NAME: &'static str = "pawnocchio_chonker_pp";

const SUPERBATCHES: usize = 3000;
const L1: usize = 4096;
const L2: usize = 128;
const L3: usize = 256;
const SCALE: i32 = 400;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;
const INPUT_BUCKETS: usize = 4;
const OUTPUT_BUCKETS: usize = 8;

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

static FILTER_TOTAL: AtomicU64 = AtomicU64::new(0);
static FILTER_KEPT: AtomicU64 = AtomicU64::new(0);
static FILTER_VIRI_REJECTED: AtomicU64 = AtomicU64::new(0);
static FILTER_PIECE_COUNT_REJECTED: AtomicU64 = AtomicU64::new(0);
const FILTER_STATS_SAMPLE_RATE: f64 = 0.01;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  0,  1,  1,
     2,  2,  2,  2,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
     3,  3,  3,  3,
    //  0,  1,  2,  3,
    //  4,  5,  6,  7,
    //  8,  8,  9,  9,
    // 10, 10, 11, 11,
    // 12, 12, 13, 13,
    // 12, 12, 13, 13,
    // 14, 14, 15, 15,
    // 14, 14, 15, 15,
];

fn piece_count_acceptance(board: &Board) -> f64 {
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    static PIECE_COUNT_STATS: [AtomicU64; 33] = zeroed();
    static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

    let pc = board.pieces.occupied().count() as usize;
    let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let frequency = count as f64 / total as f64;

    // Calculate the acceptance probability for this piece count
    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    const default_viri_filter: Filter = Filter {
        min_ply: 16,
        min_pieces: 4,
        filter_tactical: true,
        filter_check: true,
        filter_castling: true,
        max_eval: 10000,
        max_eval_incorrectness: 2500,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.15,

        wdl_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.0,
    };
    let mut rng = rng();
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    let sample_stats = rng.random_bool(FILTER_STATS_SAMPLE_RATE);

    if sample_stats {
        FILTER_TOTAL.fetch_add(1, Ordering::Relaxed);
    }

    if default_viri_filter.should_filter(mv, eval as i32, board, wdl, &mut rng) {
        if sample_stats {
            FILTER_VIRI_REJECTED.fetch_add(1, Ordering::Relaxed);
        }
        return false;
    }

    if !rng.random_bool(piece_count_acceptance(board)) {
        if sample_stats {
            FILTER_PIECE_COUNT_REJECTED.fetch_add(1, Ordering::Relaxed);
        }
        return false;
    }

    if sample_stats {
        FILTER_KEPT.fetch_add(1, Ordering::Relaxed);
    }

    true
}

fn print_filter_stats() {
    let sampled_total = FILTER_TOTAL.load(Ordering::Relaxed);
    let kept = FILTER_KEPT.load(Ordering::Relaxed);
    let viri_rejected = FILTER_VIRI_REJECTED.load(Ordering::Relaxed);
    let piece_count_rejected = FILTER_PIECE_COUNT_REJECTED.load(Ordering::Relaxed);

    let pct = |count| 100.0 * count as f64 / sampled_total.max(1) as f64;

    println!("kept: {:.2}%", pct(kept));
    println!("viri rejected: {:.2}%", pct(viri_rejected));
    println!("piece-count rejected: {:.2}%", pct(piece_count_rejected));
}

#[path = "pawn_pawn_masked.rs"]
mod inputs;

use inputs::pawn_pawn_inputs;

fn main() {
    let inputs = pawn_pawn_inputs::PawnPawnInputs::new(BUCKET_LAYOUT, pawn_pawn_inputs::three_file_band_mask());

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Optimiser::default())
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|_, weights| {
                    let pp_threats =
                        pawn_pawn_inputs::PawnPawnInputs::TOTAL_PAIRS + pawn_pawn_inputs::PawnPawnInputs::TOTAL_THREATS;
                    let shared = weights[pp_threats * L1..(pp_threats + 768) * L1].repeat(INPUT_BUCKETS);
                    let bucketed = &weights[(pp_threats + 768) * L1..];
                    bucketed.iter().zip(shared).map(|(&a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(Q0),
            SavedFormat::id("l0w")
                .transform(|_, weights| {
                    let pp_threats =
                        pawn_pawn_inputs::PawnPawnInputs::TOTAL_PAIRS + pawn_pawn_inputs::PawnPawnInputs::TOTAL_THREATS;
                    let clip = i8::MAX as f32 / Q0 as f32;
                    println!(
                        "{} {}",
                        weights[0..pp_threats * L1]
                            .iter()
                            .copied()
                            .map(|f| { if f.clamp(-clip, clip) != f { 1 } else { 0 } })
                            .sum::<i32>(),
                        pp_threats * L1,
                    );
                    weights[0..pp_threats * L1].iter().map(|f| f.clamp(-clip, clip)).collect()
                })
                .round()
                .quantise::<i8>(Q0),
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
            SavedFormat::id("l1b").round().quantise::<i32>(Q as i32 * 256),
            SavedFormat::id("l2w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2b").round().quantise::<i32>((Q as i32).pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l3b").round().quantise::<i32>((Q as i32).pow(4)),
        ])
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), target| {
            // input layer weights (factoriser is baked into the input feature layout)
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1);

            // output layer weights
            let l1 = builder.new_affine("l1", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_hidden.concat(ntm_hidden);

            let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().squared_error(target);

            let loss = loss + 0.005 * l0_out_norm;

            (l3_out, loss)
        });
    let l0_clip = OptimiserParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);

    let l1_clip = OptimiserParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let schedule = TrainingSchedule {
        net_id: NET_NAME.to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::Warmup {
            inner: lr::LinearDecayLR {
                initial_lr: 0.001 * f32::powi(0.3, 0),
                final_lr: 0.001 * f32::powi(0.3, 7),
                final_superbatch: SUPERBATCHES,
            },
            warmup_batches: 200,
        },
        save_rate: 25,
    };

    let settings =
        LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 1024 };

    let binpack_dataset = "/k4/vine_data/vine_43/mixed_data_big.vf";

    trainer.optimiser.load_weights_from_file("zero_filled_checkpoint");
    // trainer.load_from_checkpoint("checkpoints/pawnocchio_chonker_stage1-525");
    trainer.run(&schedule, &settings, &ViriBinpackLoader::new(binpack_dataset, 8192, 16, ViriFilter::Custom(filter)));
    //
    // print_filter_stats();

    // trainer.load_from_checkpoint("checkpoints/pawnocchio_multilayer_2048_2_stage1-1");
    // trainer.save_to_checkpoint("checkpoints/pawnocchio_multilayer_2048_stage2-200_requantise");

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
