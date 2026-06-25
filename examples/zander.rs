use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const SUPERBATCHES: usize = 40;
const HIDDEN_SIZE: usize = 16;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

const DATA_PATH: &str = "/k4/pawnocchio_data2/2026_06_14/scaled/outfile_pp_chonked_7000nodes_114c09f6ccd24290b5b7b68dd136c4e53b7efa229e366e498a897b54816b7930.vf";

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| (output * 2.0).sigmoid().squared_error(target))
        .build(|builder, stm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", HIDDEN_SIZE, 1);

            let hidden = l0.forward(stm_inputs).relu();
            l1.forward(hidden)
        });

    let schedule = TrainingSchedule {
        net_id: "zander".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::LinearDecayLR { initial_lr: 1e-4, final_lr: 1e-6, final_superbatch: SUPERBATCHES },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let data_loader = {
        use loader::viribinpack::{Filter, ViriBinpackLoader, ViriFilter};

        let buffer_size_mb = 1024;
        let threads = 4;
        let filter = ViriFilter::Builtin(Filter::default());

        ViriBinpackLoader::new(DATA_PATH, buffer_size_mb, threads, filter)
    };

    trainer.run(&schedule, &settings, &data_loader);
}
