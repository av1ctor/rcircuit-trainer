
mod sdlrenderer;
mod replaybuffer;
mod dqn;
mod agent;
mod baseagent;
mod dqnagent;
mod ddqnagent;
mod loaders;

use std::collections::HashSet;
use std::f32::consts::PI;
use agent::Agent;
use baseagent::BaseAgentOptions;
use ddqnagent::DdqnAgent;
use bot_engine_gltf::env::Env;
use bot_engine_gltf::game::{Game, GameOptions, ObjectFOV};
use dqn::DqnOptions;
use sdlrenderer::SdlRenderer;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use sdl2::keyboard::Scancode;
use tch::{Tensor, Device};

const LEARNING: bool = true;
const NUM_PLAYERS: usize = if LEARNING {1} else {4};
const SCREEN_WIDTH: u32 = 1500;
const SCREEN_HEIGHT: u32 = 1000;
const ACCEL_FACTOR: f32 = if LEARNING {10.0} else {1.0};
const FPS: f32 = 60.0 * ACCEL_FACTOR; // frames per sec
const TPS: f32 = 30.0 * ACCEL_FACTOR; // thoughts per sec
const STEPS_PER_THOUGHT: u32 = ((FPS / TPS) * 1000.0) as u32;
const DELTA_TIME: f32 = if LEARNING {1.0} else {60.0 / FPS};
const RAYS: usize = 15;
const FOV: f32 = 120.0 * (PI / 180.0);
const PLAYER_RADIUS: f32 = 0.5;

const REPLAY_BUFFER_CAPACITY: usize = 15_000;
const TRAINING_BATCH_SIZE: usize = 32;
const MAX_EPISODES: usize = 80;
const LEARNING_RATE: f64 = 0.0005;
const REPLACE_TARGET_CNT: usize = 128;
const GAMMA: f64 = 0.99;
const EPSILON: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 5e-5;
const NUM_HIDDEN: usize = 32;
const MAX_STEPS: usize = 15000;

fn main(
) -> Result<(), String> {
    let path = format!("./weights/{}rays-{}hidden-{}fps-{}tps-{}epsilon.safetensors", RAYS, NUM_HIDDEN, FPS / ACCEL_FACTOR, TPS / ACCEL_FACTOR, EPSILON_MIN);
    let mut rng = StdRng::seed_from_u64(1234);
    let mut _pov = 0;

    let (level_bytes, navmesh_bytes) = if let Ok(level_bytes) = std::fs::read("./maps/level-0/map.glb") {
        if let Ok(navmesh_bytes) = std::fs::read("./maps/level-0/navmesh.bin") {
            (level_bytes, navmesh_bytes)
        }
        else {
            panic!("Error loading level-0's navmesh");
        }
    }
    else {
        panic!("Error loading level-0's model");
    };

    let (mut renderer, mut event_loop) = SdlRenderer::new(SCREEN_WIDTH, SCREEN_HEIGHT)
        .expect("Renderer initialization failed");

    let mut game = Game::new(
        &level_bytes.to_vec(), 
        &navmesh_bytes.to_vec(),
        GameOptions {
            fps: FPS,
            fov: ObjectFOV { 
                rays: RAYS, 
                angle: FOV, 
                max_dist: 5.0,
            },
            num_players: NUM_PLAYERS,
            min_keys: 4,
            player_radius: PLAYER_RADIUS,
            debug: true,
            training: LEARNING,
            max_steps: MAX_STEPS,
        },
        &mut renderer
    );

    let options = BaseAgentOptions { 
        dqn: DqnOptions {
            learning_rate: LEARNING_RATE,
            n_actions: game.get_num_actions(),
            n_obs: game.get_num_obs(),
            n_hidden: NUM_HIDDEN
        },
        gamma: GAMMA,
        epsilon: if LEARNING {EPSILON} else {EPSILON_MIN},
        epsilon_dec: if LEARNING {EPSILON_DECAY} else {0.0},
        epsilon_min: if LEARNING {EPSILON_MIN} else {EPSILON_MIN},
        batch_size: TRAINING_BATCH_SIZE,
        replay_buffer_capacity: REPLAY_BUFFER_CAPACITY,
        replace_target_cnt: REPLACE_TARGET_CNT,
    };
    
    let mut agent = DdqnAgent::new(&options);

    if !LEARNING {
        agent.load(path.to_string())?;
    }

    let mut best_win_rate = 0.0;
    let mut sample = 10;
    let mut results: Vec<usize> = vec![];

    let mut eps = 0;
    'main: while eps < MAX_EPISODES {
        let mut score = 0.0f32;

        let mut steps = 0;
        let mut running = true;
        let mut thought_step_counter = 0u32;

        let mut obs_vec = vec![]; 
        let mut obs_ts_vec = vec![];
        
        let obs = &game.reset(&mut rng);
        for i in 0..game.options.num_players {
            obs_vec.push(&obs[i]);
            obs_ts_vec.push(Tensor::from_slice(&obs[i]).to(Device::Cuda(0)));
        }

        while running {
            for event in event_loop.poll_iter() {
                match event {
                    sdl2::event::Event::Quit { .. } => {
                        break 'main;
                    },
                    sdl2::event::Event::KeyDown {scancode: Some(scancode), ..} => {
                        if scancode == Scancode::Escape {
                            break 'main;
                        }
                        else if scancode as i32 >= Scancode::Num1 as i32 && scancode as i32 <= Scancode::Num9 as i32 {
                            _pov = ((scancode as i32 - Scancode::Num1 as i32) as usize) % game.options.num_players;
                        }
                    },
                    _ => {}
                }
            }

            thought_step_counter += 1 * 1000;
            let do_think = thought_step_counter >= STEPS_PER_THOUGHT;

            let actions = if do_think  { 
                thought_step_counter = 0;
                
                let mut actions = vec![0i64; game.options.num_players];

                let players_idx = gen_players_idx(&mut game, &mut rng);
                
                for i in 0..game.options.num_players {
                    let p = players_idx[i];
                    let action_ts = agent.choose_action(&obs_ts_vec[p], &mut rng)
                        .unwrap_or(Tensor::from(game.random_action(&mut rng) as i64)
                            .to(Device::Cuda(0)));

                    let mut action = vec![0i64; 1];
                    action_ts.copy_data(&mut action, 1);
                    actions[p] = action[0];
                }

                actions.iter()
                    .map(|v| Some(v.clone() as usize))
                    .collect::<Vec<Option<usize>>>()
            } else {
                vec![None; game.options.num_players]
            };

            let step = game.step(&actions, DELTA_TIME);
            steps += 1;

            running = !step.terminated && !step.timedout;
            if !running {
                results.push(if step.winner.is_some() {1} else {0});
            }

            if LEARNING {
                score += step.players[0].reward;
            }

            for p in 0..step.players.len() {
                let obs = &step.players[p].obs;
                let next_obs_ts = Tensor::from_slice(obs).to(Device::Cuda(0));

                if LEARNING && p == 0 && do_think {
                    let action_ts = Tensor::from_slice(&[actions[0].unwrap() as i64])
                        .to(Device::Cuda(0));
                    agent.store_transition(
                        &obs_ts_vec[0], 
                        &action_ts, 
                        &step.players[0].reward.into(), 
                        &next_obs_ts, 
                        &step.terminated.into(),
                        &step.timedout.into()
                    );

                    agent.learn(&mut rng);
                }

                obs_ts_vec[p] = next_obs_ts;
            }

            if steps % 60 == 0 {
                game.render();
            }
        }

        let win_rate = results.iter().rev().take(sample).sum::<usize>() as f32 / (sample as f32);

        if win_rate > best_win_rate {
            best_win_rate = win_rate;
            if LEARNING {
                if best_win_rate >= 0.9 {
                    if agent.get_epsilon() <= EPSILON_MIN {
                        agent.save(path.to_string())?;
                    }
                    if best_win_rate == 1.0 {
                        sample *= 2;
                        best_win_rate /= 2.0;
                    }
                }
            }
        }

        println!("ep: {}, steps: {}, score: {:.0}, epsilon: {:.2}, win rate: {:.1}%, best rate: {:.1}%", 
            eps, steps, score, agent.get_epsilon(), win_rate * 100.0, best_win_rate * 100.0);

        eps += 1;
    }

    Ok(())
}

fn gen_players_idx(
    game: &mut Game,
    rng: &mut StdRng
) -> Vec<usize> {
    let mut indexes = HashSet::new();

    while indexes.len() < game.options.num_players {
        indexes.insert(rng.gen_range(0..game.options.num_players));
    }

    indexes.into_iter().collect()
}