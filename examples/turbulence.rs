use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{inputs, loader, outputs, Loss, TrainerBuilder},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

const HIDDEN_SIZE: usize = 1024;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
const SUPERBATCHES: usize = 400;
const SAVE_RATE: usize = 10;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::ChessBucketsMirrored::default())
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .round_in_quantisation()
        .build();

    let schedule = TrainingSchedule {
        net_id: "turbulence".to_string(),
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
                final_lr: 0.001 * f32::powi(0.3, 3),
                final_superbatch: SUPERBATCHES,
            },
            warmup_batches: 200,
        },
        save_rate: SAVE_RATE,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&[
        "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/turb_data/shuffled_06_27_1.5b",
    ]);

    trainer.run(&schedule, &settings, &data_loader);
}
