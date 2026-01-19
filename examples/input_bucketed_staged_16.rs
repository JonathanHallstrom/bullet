use std::{ffi::OsString, u32};

use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{inputs, loader, outputs, Loss, TrainerBuilder},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::loader::ViriBinpackLoader,
};
use viriformat::dataformat::Filter;

const SUPERBATCHES_STAGE0: usize = 200;
const SUPERBATCHES_STAGE1: usize = 400;
const SUPERBATCHES_STAGE2: usize = 200;
const HIDDEN_SIZE: usize = 1280;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
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
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::ChessBucketsMirroredFactorised::new(BUCKET_LAYOUT))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();
    let stage0_schedule: TrainingSchedule<lr::Warmup<lr::CosineDecayLR>, wdl::LinearWDL> = TrainingSchedule {
        net_id: "input_bucketed_1280_16_stage0".to_string(),
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
        net_id: "input_bucketed_1280_16_stage1".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 80,
            end_superbatch: SUPERBATCHES_STAGE1,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.3, end: 0.6 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 3),
                final_superbatch: SUPERBATCHES_STAGE1,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };
    let stage2_schedule = TrainingSchedule {
        net_id: "input_bucketed_1280_16_stage2".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES_STAGE2,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001 * f32::powi(0.3, 2),
                final_lr: 0.001 * f32::powi(0.3, 4),
                final_superbatch: SUPERBATCHES_STAGE2,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    // let filter = Filter {
    //     min_ply: 0,
    //     min_pieces: 4,
    //     max_eval: 15000,
    //     filter_tactical: true,
    //     filter_check: true,
    //     filter_castling: false,
    //     max_eval_incorrectness: u32::MAX,
    //     ..Default::default(),
    // };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_stage2-200");
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_16_stage0-190");
    // trainer.run(
    //     &stage0_schedule,
    //     &settings,
    //     &loader::DirectSequentialDataLoader::new(&[
    //         "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/net17_18_19.bin",
    //     ]),
    // );
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_16_stage1-80");
    // trainer.run(
    //     &stage1_schedule,
    //     &settings,
    //     &loader::ViriBinpackLoader::new(
    //         "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/outfile_combined_05_26.vf",
    //         8192,
    //         8,
    //         filter.clone(),
    //     ),
    //     // &loader::DirectSequentialDataLoader::new(&[
    //     //     "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/net17_18_19.bin",
    //     // ]),
    // );
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_stage1-600");
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_stage2-200");
    trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_16_stage1-400");
    // trainer.set_wdl_adjust(|pos: &bulletformat::ChessBoard, wdl| {
    //     let tmp = (pos.occ.count_ones().saturating_sub(4) as f32 / 16.0f32).powi(2);
    //     let m = 1.0f32 - tmp.tanh();
    //     // (1.0f32 - wdl) * m + wdl
    //     m * wdl + (1.0f32 - m) * 0.4
    // });
    // trainer.run(&stage2_schedule, &settings, &loader::DirectSequentialDataLoader::new(&["data/shuffled_05_10.bin"]));

    trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_16_stage2-200");
    // trainer.load_from_checkpoint("checkpoints/input_bucketed_1280_16_stage2-170");
    trainer.run(
        &stage2_schedule,
        &settings,
        &loader::ViriBinpackLoader::new(
            "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/outfile_combined_05_24.vf",
            8192,
            8,
            Filter {
                min_ply: 0,
                min_pieces: 4,
                max_eval: 15000,
                filter_tactical: true,
                filter_check: true,
                filter_castling: false,
                max_eval_incorrectness: u32::MAX,
                ..Default::default(),
            },
        ),
        // &loader::DirectSequentialDataLoader::new(&[
        //     "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/net17_18_19.bin",
        // ]),
    );

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
