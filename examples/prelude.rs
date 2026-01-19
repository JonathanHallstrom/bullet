// This the main config for training the NNUEs used by Prelude
use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{
            inputs, loader, outputs, Loss, TrainerBuilder,
        },
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};
use viriformat::dataformat::Filter;

const HIDDEN_SIZE: usize = 1024;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;
const SUPERBATCHES: usize = 500;
const SAVE_RATE: usize = 50;
const OUTPUT_BUCKETS: usize = 8;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
	    .input(inputs::Chess768)
        .output_buckets(outputs::MaterialCount::<OUTPUT_BUCKETS>)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "Prelude_09".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.35 },
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

    let settings = LocalSettings {
        threads: 8,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 512
    };

	// Default filter (from Viri)
    let filter = Filter::default();

    // let non_relative_paths = std::fs::read_dir("./data").unwrap().into_iter().flatten().map(|f|  f.file_name()).collect::<Vec<_>>();
    // let actual_paths = non_relative_paths.iter().map(|f: &std::ffi::OsString| f.to_str()).flatten().map(|s| "./data/".to_owned() + s).collect::<Vec<_>>();
    // let slices = actual_paths.iter().map(|f| f.as_str()).collect::<Vec<_>>();
    // let data_loader = loader::DirectSequentialDataLoader::new(&slices);
    // let data_loader = loader::ViriBinpackLoader::new("data/outfile_combined.vf", 1024, 8, filter);

    trainer.run(&schedule, &settings, &loader::DirectSequentialDataLoader::new(&[
        "/media/jonathanhallstrom/64a18cc9-6680-4f1b-a09f-56b812251151/pawnocchio_data_backup/shuffled_05_15.bin",
    ]),);
}
