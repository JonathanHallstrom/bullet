
use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{inputs, loader, outputs, Loss, TrainerBuilder},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};
use viriformat::dataformat::Filter;

const HIDDEN_SIZE: usize = 384;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
const SUPERBATCHES: usize = 300;
const SAVE_RATE: usize = 10;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::Chess768::default())
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::CReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "stockdory".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 4),
                final_superbatch: SUPERBATCHES,
            },
            warmup_batches: 200,
        },
        save_rate: SAVE_RATE,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let filter = Filter {
        min_ply: 0,
        min_pieces: 4,
        max_eval: 15000,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: 400,
        random_fen_skipping: false,
        random_fen_skip_probability: 0.0,
        wld_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        normalise_to_pawn_value: 100,
        wdl_heuristic_scale: 0.0,
    };
    trainer.run(&schedule, &settings, 
        &loader::ViriBinpackLoader::new(
            "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/outfile_combined_06_02.vf",
            8192,
            8,
            filter.clone(),
        ),
    );
}
