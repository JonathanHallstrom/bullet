use std::ffi::OsString;

use bullet_lib::{
     nn::{optimiser, Activation}, trainer::{
        default::{
            inputs, loader, outputs, Loss, TrainerBuilder,
        },
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    }
};

const SUPERBATCHES: usize = 800;
const HIDDEN_SIZE: usize = 1024;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  4,  5,  5,
     6,  6,  7,  7,
     8,  8,  9,  9,
    10, 10, 10, 10,
    10, 10, 10, 10,
    11, 11, 11, 11,
    11, 11, 11, 11,
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
    let schedule = TrainingSchedule {
        net_id: "input_bucketed".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        // wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        wdl_scheduler: wdl::LinearWDL { start: 0.0, end: 0.4 },
        // lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.995, step: 1 } ,
        // lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 8 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 3),
                final_superbatch: SUPERBATCHES,
            },
            warmup_batches: 200,
        },
        save_rate: 10,
    };
    
    trainer.set_optimiser_params(optimiser::AdamWParams::default());
    
    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };
    

    // load everything in ./data
    // yeah its kinda not good
    let non_relative_paths = std::fs::read_dir("./data").unwrap().into_iter().flatten().map(|f|  f.file_name()).collect::<Vec<_>>();
    let actual_paths = non_relative_paths.iter().map(|f: &OsString| f.to_str()).flatten().map(|s| "./data/".to_owned() + s).collect::<Vec<_>>();
    let slices = actual_paths.iter().map(|f| f.as_str()).collect::<Vec<_>>();

    let data_loader = loader::DirectSequentialDataLoader::new(&slices);
    
    trainer.run(&schedule, &settings, &data_loader);
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
