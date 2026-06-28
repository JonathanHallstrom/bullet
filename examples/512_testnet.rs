use std::cell::{Cell, RefCell};

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::MaterialCount,
    },
    nn::{
        Shape,
        optimiser::{AdamW, AdamWParams, Ranger, RangerParams},
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
use rand::{Rng, rng, seq::SliceRandom};
use std::sync::atomic::{AtomicU64, Ordering};
use viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::{Filter, WDL},
};
type Optimiser = AdamW;
const NET_NAME: &'static str = "512_testnet_unscaled";

const SUPERBATCHES_STAGE1: usize = 200;
const HIDDEN_SIZE: usize = 512;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
const NUM_OUTPUT_BUCKETS: usize = 8;

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

    thread_local! {
        static PIECE_COUNT_STATS: RefCell<[u64; 33]> = RefCell::new([0; 33]);
        static PIECE_COUNT_TOTAL: RefCell<u64> = RefCell::new(0);
    }

    let pc = board.pieces.occupied().count() as usize;

    let (count, total) = PIECE_COUNT_STATS.with(|stats_cell| {
        PIECE_COUNT_TOTAL.with(|total_cell| {
            let mut stats = stats_cell.borrow_mut();
            let mut total = total_cell.borrow_mut();

            // Update stats
            stats[pc] += 1;
            *total += 1;

            (stats[pc], *total)
        })
    });

    let frequency = count as f64 / total as f64;

    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let default_viri_filter = Filter {
        min_ply: 16,
        min_pieces: 4,
        max_eval: 5000,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: 1024,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.25,
        wdl_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        wdl_heuristic_scale: 0.0,
        material_min: 0,
        material_max: 32,
        mom_target: 0,
    };
    let mut rng = rng();
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    !default_viri_filter.should_filter(mv, eval as i32, board, wdl, &mut rng)
        && rng.random_bool(piece_count_acceptance(board))
}

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Optimiser::default())
        .inputs(ChessBucketsMirrored::default())
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").quantise::<i16>(QA),
            SavedFormat::id("l0b").quantise::<i16>(QA),
            SavedFormat::id("l1w").quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: NET_NAME.to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384 * 8,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES_STAGE1,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 4),
                final_superbatch: SUPERBATCHES_STAGE1,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };

    let settings =
        LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 1024 };

    // let viriformat_dataset = "/home/jonathanhallstrom/dev/rust/bullet/2985/pawnocchio_data_for_vine_comparison.vf";
    // let viriformat_dataset = "/home/jonathanhallstrom/dev/rust/bullet/vine_dataset32.vf";
    // let viriformat_dataset = "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/vine_data/dataset_33_relabel/vine_dataset33_relabelled.vf";
    // let binpack_dataset = "/home/jonathanhallstrom/dev/rust/bullet/vine_37/vine_dataset37_partial_relabelled.vf";
    let binpack_dataset = "/home/jonathanhallstrom/dev/rust/bullet/vine_38/vine_38_10m.vf_relabeled";
    // let binpack_dataset = "/home/jonathanhallstrom/dev/rust/bullet/vine_39/output1.bin_relabeled";
    let binpack_dataset = "/home/jonathanhallstrom/dev/rust/bullet/vine_40/vine_40_10m.vf_relabeled";
    let binpack_dataset = "/home/jonathanhallstrom/dev/rust/bullet/vine_42/vine_42_10m.vf_relabeled";

    let dataset = |g| {
        let paths = glob::glob(g).expect("successfully found dataset").map(|f| f.unwrap()).collect::<Vec<_>>();
        let mut filenames = paths.iter().map(|f| f.to_str().unwrap().to_owned()).collect::<Vec<_>>();
        filenames.shuffle(&mut rng());
        filenames
    };
    let binpack_dataset = dataset("/k4/pawnocchio_data2/2026_06_14/unscaled/*.vf");
    let loader = |dataset: &[String]| {
        let strs: Vec<&str> = dataset.iter().map(|s| s.as_str()).collect();
        ViriBinpackLoader::new_concat_multiple(&strs, 8192, 16, ViriFilter::Custom(filter))
    };

    trainer.run(&schedule, &settings, &loader(&binpack_dataset));

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
