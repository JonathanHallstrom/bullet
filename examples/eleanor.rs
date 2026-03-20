use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

fn main() {
    // hyperparams to fiddle with
    let hl_size = 1536;
    let dataset_path = "/k4/eleanor_data/2026_03_14/combined.binpack";
    let initial_lr_s1 = 0.001 * 0.3f32.powi(0);
    let final_lr_s1 = 0.001 * 0.3f32.powi(7);
    let initial_lr_s2 = 0.001 * 0.3f32.powi(4);
    let final_lr_s2 = 0.001 * 0.3f32.powi(8);
    let superbatches_s1 = 600;
    let superbatches_s2 = 200;
    let initial_wdl_s1 = 0.3;
    let final_wdl_s1 = 0.3;
    let wdl_constant_s2 = 0.8;

    const NUM_OUTPUT_BUCKETS: usize = 8;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::default())
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").quantise::<i16>(255),
            SavedFormat::id("l0b").quantise::<i16>(255),
            // we want to save output-bucketed weights in a format
            // that is suitable for fast cpu inference
            SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // weights
            let l0 = builder.new_affine("l0", 768, hl_size);
            let l1 = builder.new_affine("l1", 2 * hl_size, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    let net_id = "eleanor4_1536hl";
    let schedule_s1 = TrainingSchedule {
        net_id: net_id.to_string() + "_stage1",
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384 * 8,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: superbatches_s1,
        },
        wdl_scheduler: wdl::LinearWDL { start: initial_wdl_s1, end: final_wdl_s1 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: initial_lr_s1,
            final_lr: final_lr_s1,
            final_superbatch: superbatches_s1,
        },
        save_rate: 10,
    };
    let schedule_s2 = TrainingSchedule {
        net_id: net_id.to_string() + "_stage2",
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384 * 8,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: superbatches_s2,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_constant_s2 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: initial_lr_s2,
            final_lr: final_lr_s2,
            final_superbatch: superbatches_s2,
        },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 24, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader =
        loader::ViriBinpackLoader::new(dataset_path, 1024 * 8, 24, viriformat::dataformat::Filter::default());

    trainer.run(&schedule_s1, &settings, &dataloader);
    // trainer.load_from_checkpoint("checkpoints/eleanor_1536hl_stage1-600/");
    trainer.run(&schedule_s2, &settings, &dataloader);
}
