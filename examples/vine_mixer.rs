use std::u32;

use bullet_lib::{
    game::inputs::{self},
    nn::{InitSettings, Shape, optimiser},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

use bulletformat::ChessBoard;
use montyformat::chess::{Attacks, Piece, Side, consts::IN_BETWEEN};
use viriformat::dataformat::Filter;

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        return 3072;
    }
    fn max_active(&self) -> usize {
        return 32;
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        return "3072".to_owned();
    }
    fn description(&self) -> String {
        return "SuperSimpleThreats".to_owned();
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

fn map_features<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let occ = bbs[0] | bbs[1];
    let mut threats: [u64; 2] = [0; 2];
    let mut pinned: [u64; 2] = [0; 2];

    for side in [Side::WHITE, Side::BLACK] {
        let us = bbs[side];
        let our_king_idx = (bbs[Piece::KING] & us).trailing_zeros() as usize;
        let bishops = bbs[Piece::BISHOP];
        let rooks = bbs[Piece::ROOK];
        let queens = bbs[Piece::QUEEN];

        let possible_pinners_rook = Attacks::xray_rook(our_king_idx, occ, us) & (rooks | queens) & !us;
        let possible_pinners_bishop = Attacks::xray_bishop(our_king_idx, occ, us) & (bishops | queens) & !us;

        map_bb(possible_pinners_bishop | possible_pinners_rook, |pinner| {
            let between = IN_BETWEEN[our_king_idx][pinner];
            if (between & us).count_ones() == 1 {
                pinned[side] |= between;
            }
        });
    }

    for side in [Side::WHITE, Side::BLACK] {
        let our_king_idx = (bbs[Piece::KING] & bbs[side]).trailing_zeros() as usize;
        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let mut cur_threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                if pinned[side] & 1 << sq != 0 {
                    cur_threats &= IN_BETWEEN[our_king_idx][sq];
                }
                threats[side] |= cur_threats;
            });
        }
    }

    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let mut feat = [0, 384][side] + 64 * (piece - 2) + sq;
                let bit = 1 << sq;
                if threats[side ^ 1] & bit > 0 {
                    feat += 768;
                }
                if threats[side] & bit > 0 {
                    feat += 768 * 2;
                }
                f(feat);
            });
        }
    }
}

const SUPERBATCHES: usize = 3000;
const DIM_GRAIN: usize = 16;
const MIXER_D1: usize = 1 * DIM_GRAIN;
const MIXER_D2: usize = 16 * DIM_GRAIN;
const MIXER_UP1: usize = MIXER_D1;
const MIXER_UP2: usize = MIXER_D2;
const NUM_LAYERS: usize = 2;
const L1: usize = MIXER_D1 * MIXER_D2;
const SCALE: i32 = 400;
const QA: i16 = 255;

fn main() {
    let mut save_format =
        vec![SavedFormat::id("l0w").round().quantise::<i16>(QA), SavedFormat::id("l0b").round().quantise::<i16>(QA)];

    for i in 1..=NUM_LAYERS {
        save_format.push(SavedFormat::id(format!("wl{i}_up")));
        save_format.push(SavedFormat::id(format!("wl{i}_down")));
        save_format.push(SavedFormat::id(format!("wr{i}_up")));
        save_format.push(SavedFormat::id(format!("wr{i}_down")));
    }
    save_format.append(vec![SavedFormat::id("value_headw"), SavedFormat::id("value_headb")]);

    let mut trainer = ValueTrainerBuilder::default()
        .single_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ThreatInputs::default())
        .save_format(&save_format)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs| {
            let l0 = builder.new_affine("l0", 3072, L1);
            let x_flat = l0.forward(stm_inputs).crelu();

            let mut x = x_flat.reshape(Shape::new(MIXER_D1, MIXER_D2));

            let init_up_d1 = InitSettings::Normal { mean: 0.0, stdev: (2.0 / MIXER_D1 as f32).sqrt() };
            let init_down_d1 = InitSettings::Normal { mean: 0.0, stdev: (2.0 / MIXER_UP1 as f32).sqrt() };
            let init_up_d2 = InitSettings::Normal { mean: 0.0, stdev: (2.0 / MIXER_D2 as f32).sqrt() };
            let init_down_d2 = InitSettings::Normal { mean: 0.0, stdev: (2.0 / MIXER_UP2 as f32).sqrt() };

            for i in 1..=NUM_LAYERS {
                let wl_up = builder.new_weights(format!("wl{i}_up"), Shape::new(MIXER_UP1, MIXER_D1), init_up_d1);
                let wl_down = builder.new_weights(format!("wl{i}_down"), Shape::new(MIXER_D1, MIXER_UP1), init_down_d1);
                let left_mix = wl_down.matmul(wl_up.matmul(x).crelu());
                x = x + left_mix;

                let wr_up = builder.new_weights(format!("wr{i}_up"), Shape::new(MIXER_D2, MIXER_UP2), init_up_d2);
                let wr_down = builder.new_weights(format!("wr{i}_down"), Shape::new(MIXER_UP2, MIXER_D2), init_down_d2);
                let right_mix = x.matmul(wr_up).crelu().matmul(wr_down);
                x = x + right_mix;
            }

            let x_final = x.reshape(Shape::new(L1, 1));

            let v_head = builder.new_affine("value_head", L1, 1);
            v_head.forward(x_final)
        });

    let schedule = TrainingSchedule {
        net_id: "vine_mixer_64x64".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384 / 4,
            batches_per_superbatch: 6104 * 4,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::Warmup { inner: wdl::ConstantWDL { value: 1.0 }, warmup_batches: 2000 },
        lr_scheduler: lr::Warmup {
            inner: lr::LinearDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * f32::powi(0.3, 7),
                final_superbatch: SUPERBATCHES,
            },
            warmup_batches: 2000,
        },
        save_rate: 100,
    };

    let settings =
        LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 1024 };

    let data_loader = loader::ViriBinpackLoader::new(
        "/k4/vine_data/vine_43/vine_43_adj.vf",
        16384,
        16,
        Filter {
            min_ply: 0,
            min_pieces: 0,
            max_eval: 15000,
            filter_tactical: false,
            filter_check: false,
            filter_castling: false,
            max_eval_incorrectness: u32::MAX,
            random_fen_skipping: false,
            random_fen_skip_probability: 0.0,
            wdl_filtered: false,
            wdl_model_params_a: [0.0; 4],
            wdl_model_params_b: [0.0; 4],
            wdl_heuristic_scale: 0.0,
            material_min: 0,
            material_max: 32,
            mom_target: 0,
        },
    );

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "1k6/3r4/8/8/8/2r5/3P4/3KR3 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
    ] {
        let eval = 400.0 * trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", eval);
    }
}
