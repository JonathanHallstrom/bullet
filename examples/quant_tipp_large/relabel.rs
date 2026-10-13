mod inputs;
mod net;

use std::{
    collections::VecDeque,
    env,
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Write},
    sync::mpsc::{self, Receiver, Sender, SyncSender},
    thread,
    time::Instant,
};

use bullet_lib::{game::formats::bulletformat::ChessBoard, trainer::schedule::wdl};
use bullet_trainer::{
    model::{ModelEvaluator, ModelWeights},
    run::{DefaultDevice, HostPool, Step},
};
use viriformat::{chess::piece::Colour, dataformat::Game};

const BATCH_SIZE: usize = 16384;
const QUEUE_DEPTH: usize = 2;
const MAP_THREADS: u8 = 8;
const SCALE: f32 = 400.0;
const EVAL_CLAMP: i32 = 32767;

fn gpu_error(err: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{err:?}"))
}

fn stored_eval(output: f32, black_to_move: bool) -> i16 {
    let score = (SCALE * output).round() as i32;
    let score = if black_to_move { -score } else { score };
    score.clamp(-EVAL_CLAMP, EVAL_CLAMP) as i16
}

fn read_games(
    mut reader: impl BufRead,
    games: Sender<Game>,
    batches: SyncSender<(Vec<ChessBoard>, usize)>,
) -> io::Result<()> {
    let mut boards = Vec::new();

    loop {
        // break if we hit the end of the file
        if reader.fill_buf()?.is_empty() {
            break;
        }
        let game = Game::deserialise_from(&mut reader, Vec::new())?;
        game.splat_to_bulletformat_with_filter_callback(
            |board| {
                boards.push(board);
                Ok(())
            },
            |_, _, _, _, _| false,
        )
        .map_err(io::Error::other)?;
        if games.send(game).is_err() {
            return Ok(());
        }
        while boards.len() >= BATCH_SIZE {
            if batches.send((boards.drain(..BATCH_SIZE).collect(), 0)).is_err() {
                return Ok(());
            }
        }
    }

    if !boards.is_empty() {
        let padding = BATCH_SIZE - boards.len();
        boards.resize(BATCH_SIZE, Default::default());
        let _ = batches.send((boards, padding));
    }

    Ok(())
}

fn relabel<B>(
    batches: Receiver<(B, usize)>,
    games: Receiver<Game>,
    writer: &mut impl Write,
    mut evaluate: impl FnMut(B, &mut Vec<f32>) -> io::Result<()>,
) -> io::Result<(usize, usize)> {
    let mut pending = VecDeque::new();
    let mut scores = Vec::new();
    let mut output_moves = Vec::new();
    let mut game_count = 0;
    let mut position_count = 0;

    loop {
        let batch = batches.recv().ok();
        let finished = batch.is_none();
        if let Some((batch, padding)) = batch {
            let expected = scores.len() + BATCH_SIZE;
            evaluate(batch, &mut scores)?;
            assert_eq!(scores.len(), expected);
            scores.truncate(scores.len() - padding);
            pending.extend(games.try_iter());
        } else {
            pending.extend(games.iter());
        }

        let mut ready_games = 0;
        let mut ready_scores = 0;
        for game in &pending {
            if ready_scores + game.len() > scores.len() {
                break;
            }
            ready_games += 1;
            ready_scores += game.len();
        }
        {
            let mut ready = scores.drain(..ready_scores);
            for game in pending.drain(..ready_games) {
                let mut black_to_move = game.initial_position().turn() == Colour::Black;
                output_moves.clear();
                let mut relabelled = Game { initial_position: game.initial_position, moves: output_moves };
                for mv in game.moves() {
                    relabelled.add_move(mv, stored_eval(ready.next().unwrap(), black_to_move));
                    black_to_move = !black_to_move;
                }
                relabelled.serialise_into(writer)?;
                output_moves = relabelled.into_move_buffer();
            }
        }
        game_count += ready_games;
        position_count += ready_scores;

        if finished {
            if !pending.is_empty() || !scores.is_empty() {
                return Err(io::Error::other("positions left unscored at end of input"));
            }
            return Ok((game_count, position_count));
        }
    }
}

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let weights_path = args.next().ok_or_else(|| io::Error::other("missing weights.bin file"))?;
    let input = args.next().ok_or_else(|| io::Error::other("missing input vf file"))?;
    let output = args.next().ok_or_else(|| io::Error::other("missing output vf file"))?;

    let (model_inputs, pp, psqt, defn) = net::build();
    let mut weights = ModelWeights::zeroed(&defn);
    weights.load_from(BufReader::new(File::open(weights_path)?))?;

    let device = DefaultDevice::new(0).map_err(gpu_error)?;
    let mut evaluator = ModelEvaluator::with_batch_size(&defn, device.clone(), BATCH_SIZE).map_err(gpu_error)?;
    evaluator.load_weights(&weights).map_err(gpu_error)?;
    let mapper = inputs::make_inputs_mapper(&model_inputs, pp, psqt, wdl::ConstantWDL { value: 0.0 });
    let pool = HostPool::new(device.clone());

    let reader = BufReader::new(File::open(input)?);
    let output = OpenOptions::new().write(true).create_new(true).open(output)?;
    let mut writer = BufWriter::new(output);

    let (game_tx, game_rx) = mpsc::channel();
    let (board_tx, board_rx) = mpsc::sync_channel(QUEUE_DEPTH);
    let (batch_tx, batch_rx) = mpsc::sync_channel(QUEUE_DEPTH);

    let start = Instant::now();
    let (game_count, position_count) = thread::scope(|s| {
        let reading = s.spawn(move || read_games(reader, game_tx, board_tx));
        let mapping = s.spawn(move || -> io::Result<()> {
            for (boards, padding) in board_rx {
                let batch = mapper.map(&pool, &boards, Step::default(), MAP_THREADS).map_err(gpu_error)?;
                if batch_tx.send((batch, padding)).is_err() {
                    break;
                }
            }
            Ok(())
        });

        let result = relabel(batch_rx, game_rx, &mut writer, |batch, scores| {
            let inputs = batch.to_device(&device).map_err(gpu_error)?;
            let outputs = evaluator.evaluate(&inputs).map_err(gpu_error)?;
            let output = outputs["output"].to_host().map_err(gpu_error)?;
            scores.extend_from_slice(output.f32());
            Ok(())
        });

        reading.join().unwrap()?;
        mapping.join().unwrap()?;
        result
    })?;

    writer.flush()?;
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "relabelled {game_count} games, {position_count} positions in {elapsed:.2}s ({:.0} pos/s)",
        position_count as f64 / elapsed,
    );
    Ok(())
}
