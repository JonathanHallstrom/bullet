use std::default;

/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
There's potentially a lot of elo available by adjusting the wdl
and lr schedulers, depending on your dataset.
*/
use bullet_lib::{
    game::{
        formats::sfbinpack::{
            TrainingDataEntry,
            chess::{r#move::MoveType, piecetype::PieceType},
        },
        inputs::{self, get_num_buckets},
        outputs,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{self, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{self, ViriBinpackLoader},
    },
};

const L1: usize = 1024;
const L2: usize = 16;
const L3: usize = 32;
const SCALE: i32 = 400;
const SUPERBATCHES: usize = 400;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;
const OUTPUT_BUCKETS: usize = 8;

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        // makes `ntm_inputs` available below
        .dual_perspective()
        // standard optimiser used in NNUE
        // the default AdamW params include clipping to range [-1.98, 1.98]
        .optimiser(optimiser::AdamW)
        // basic piece-square chessboard inputs
        .inputs(inputs::ChessBucketsMirrored::default())
        .output_buckets(outputs::MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(Q0),
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
            SavedFormat::id("l1b").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l2b").round().quantise::<i32>((Q as i32).pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(Q as i32),
            SavedFormat::id("l3b").round().quantise::<i32>((Q as i32).pow(4)),
        ])
        // are in the same range
        // `target` == wdl * game_result + (1 - wdl) * sigmoid(search score in centipawns / SCALE)
        // where `wdl` is determined by `wdl_scheduler`
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        // the basic `(768 -> N)x2 -> 1` inference
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // weights
            let l0 = builder.new_affine("l0", 768, L1);

            let l1 = builder.new_affine("l1", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(hl1).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();
            // let hl2 = l1_out.screlu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            let l3_out = l3.forward(hl3).select(output_buckets);
            l3_out
        });
    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let schedule = TrainingSchedule {
        net_id: "kirill_1024_pw".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384 * 8,
            batches_per_superbatch: 6104 / 8,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 1e-3,
            final_lr: 1e-3 * 0.3f32.powi(5),
            final_superbatch: SUPERBATCHES,
        },

        save_rate: 100,
    };

    let settings = LocalSettings { threads: 16, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    // loading from a SF binpack

    let data_loader = {
        let file_path = "/k4/kirill_data/2026_04_01/combined.vf";
        let buffer_size_mb = 8192;
        let threads = 16;
        ViriBinpackLoader::new(file_path, buffer_size_mb, threads, viriformat::dataformat::Filter::default())
        // fn filter(entry: &TrainingDataEntry) -> bool {
        //     !entry.pos.is_checked(entry.pos.side_to_move())
        //         && entry.score.unsigned_abs() <= 10000
        //         && entry.mv.mtype() == MoveType::Normal
        //         && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        // }
    };

    // loading directly from a `BulletFormat` file
    // let data_loader = loader::DirectSequentialDataLoader::new(&["/k4/kirill_data/2026_01_01/pos_sanitised.bf"]);

    // trainer.load_from_checkpoint("checkpoints/kirill_1024_pw_ib-200");
    // trainer.save_to_checkpoint("checkpoints/kirill_1024_pw_ib-200");
    trainer.run(&schedule, &settings, &data_loader);
    // trainer.load_from_checkpoint("checkpoints/kirill_512_multi-200");

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
