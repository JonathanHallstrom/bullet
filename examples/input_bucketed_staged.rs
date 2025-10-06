use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{DirectSequentialDataLoader, ViriBinpackLoader},
    },
};
use viriformat::dataformat::Filter;

type Optimiser = AdamW;
type OptimiserParams = AdamWParams;
const NET_NAME: &'static str = "input_bucketed_1536_16";

const SUPERBATCHES_STAGE0: usize = 200;
const SUPERBATCHES_STAGE1: usize = 600;
const SUPERBATCHES_STAGE2: usize = 200;
const HIDDEN_SIZE: usize = 1536;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
const NUM_INPUT_BUCKETS: usize = 16;
const NUM_OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
    // 0, 0, 1, 2,
    // 3, 3, 4, 4,
    // 5, 5, 5, 5,
    // 6, 6, 6, 6,
    // 6, 6, 6, 6,
    // 7, 7, 7, 7,
    // 7, 7, 7, 7,
    // 7, 7, 7, 7,
    // 0, 1, 2, 3,
    // 4, 4, 5, 5,
    // 6, 6, 6, 6,
    // 7, 7, 7, 7,
    // 7, 7, 7, 7,
    // 7, 7, 7, 7,
    // 7, 7, 7, 7,
    // 7, 7, 7, 7,
    //  0,  1,  2,  3,
    //  4,  4,  5,  5,
    //  6,  6,  7,  7,
    //  8,  8,  9,  9,
    // 10, 10, 10, 10,
    // 10, 10, 10, 10,
    // 11, 11, 11, 11,
    // 11, 11, 11, 11,
];

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Optimiser::default())
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .add_transform(|builder, _, mut weights| {
                    let factoriser = builder.get_weights("l0f").get_dense_vals().unwrap();
                    let expanded = factoriser.repeat(NUM_INPUT_BUCKETS);

                    for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                        *i += j;
                    }

                    weights
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });
    let stricter_clipping = OptimiserParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let stage0_schedule = TrainingSchedule {
        net_id: NET_NAME.to_string() + "_stage0",
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES_STAGE0,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.0, end: 0.3 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 3),
                final_superbatch: SUPERBATCHES_STAGE0,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };
    let stage1_schedule = TrainingSchedule {
        net_id: NET_NAME.to_string() + "_stage1",
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 401,
            end_superbatch: SUPERBATCHES_STAGE1,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.3, end: 0.7 },
        lr_scheduler: lr::Warmup {
            inner: lr::LinearDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 3),
                final_superbatch: SUPERBATCHES_STAGE1,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };
    let stage2_schedule = TrainingSchedule {
        net_id: NET_NAME.to_string() + "_stage2",
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES_STAGE2,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::Warmup {
            inner: lr::ExponentialDecayLR {
                initial_lr: 0.001 * f32::powi(0.3, 3),
                final_lr: 0.001 * f32::powi(0.3, 5),
                final_superbatch: SUPERBATCHES_STAGE2,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };

    let filter = Filter {
        min_ply: 16,
        min_pieces: 4,
        filter_tactical: true,
        filter_check: true,
        filter_castling: true,
        max_eval: 10000,
        max_eval_incorrectness: 2500,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.5,

        wdl_filtered: true,
        wdl_model_params_a: [-51.91819866, 145.18809272, -166.61481017, 281.59570002],
        wdl_model_params_b: [-24.71724508, 82.92975519, -33.49186286, 52.86407201],
        ..Default::default()
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let binpack_dataset = "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/regen_0821_5-12ksn_dataset.vf";

    // trainer.run(
    //     &stage0_schedule,
    //     &settings,
    //     &DirectSequentialDataLoader::new(&[
    //         "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/net17_18_19.bin",
    //     ]),
    // );
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1536_stage0-200/");
    trainer.load_from_checkpoint("checkpoints/input_bucketed_1536_16_stage1-400/");
    trainer.run(&stage1_schedule, &settings, &ViriBinpackLoader::new(binpack_dataset, 32768, 16, filter.clone()));
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1536_stage1-600/");
    trainer.run(&stage2_schedule, &settings, &ViriBinpackLoader::new(binpack_dataset, 32768, 16, filter.clone()));

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
