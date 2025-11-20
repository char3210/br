#![allow(unused)]

use std::{collections::HashMap, fs::{read_dir, read_to_string}};

use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng, seq::SliceRandom};


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BranchDir {
    Taken,
    NotTaken,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct BranchInst {
    bits: u32,
    target: u32,
}

trait BranchPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir;
    fn update(&mut self, pc: u32, inst: BranchInst, actual: BranchDir) {}
}


struct PerfectBimodalPredictor {
    table: HashMap<u32, u8>,
    initial_value: u8,
}

impl PerfectBimodalPredictor {
    fn new(initial_value: u8) -> Self {
        PerfectBimodalPredictor {
            table: HashMap::new(),
            initial_value,
        }
    }
}

impl BranchPredictor for PerfectBimodalPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir {
        let counter = self.table.entry(pc).or_insert(self.initial_value);
        let prediction = if *counter >= 2 { BranchDir::Taken } else { BranchDir::NotTaken };

        prediction
    }

    fn update(&mut self, pc: u32, inst: BranchInst, actual: BranchDir) {
        let counter = self.table.entry(pc).or_insert(self.initial_value);
        match actual {
            BranchDir::Taken => {
                if *counter < 3 {
                    *counter += 1;
                }
            }
            BranchDir::NotTaken => {
                if *counter > 0 {
                    *counter -= 1;
                }
            }
        };
    }
}

struct PerfectGshareXorPredictor {
    table: HashMap<u32, u8>,
    global_history: u32,
    history_bits: u32,
    initial_value: u8,
}

impl PerfectGshareXorPredictor {
    fn new(history_bits: u32, initial_value: u8) -> Self {
        PerfectGshareXorPredictor {
            table: HashMap::new(),
            global_history: 0,
            history_bits,
            initial_value,
        }
    }
}
impl BranchPredictor for PerfectGshareXorPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir {
        let history_mask = (1 << self.history_bits) - 1;
        let history = self.global_history & history_mask;
        let index = pc ^ (history << (32 - self.history_bits));
        let counter = self.table.entry(index).or_insert(self.initial_value);
        let prediction = if *counter >= 2 { BranchDir::Taken } else { BranchDir::NotTaken };

        self.global_history = (self.global_history << 1) | match prediction {
            BranchDir::Taken => 1,
            BranchDir::NotTaken => 0,
        };

        prediction
    }

    fn update(&mut self, pc: u32, inst: BranchInst, actual: BranchDir) {
        let history_mask = (1 << self.history_bits) - 1;
        let history = (self.global_history >> 1) & history_mask; // hack but whatever
        let index = pc ^ (history << (32 - self.history_bits));
        let counter = self.table.entry(index).or_insert(self.initial_value);
        match actual {
            BranchDir::Taken => {
                if *counter < 3 {
                    *counter += 1;
                }
            }
            BranchDir::NotTaken => {
                if *counter > 0 {
                    *counter -= 1;
                }
            }
        };
    }
}


struct PerfectBimodalGHistPredictor {
    table: HashMap<(u32, u128), u8>,
    global_history: u128,
    history_bits: u32,
    initial_value: u8,
}

impl PerfectBimodalGHistPredictor {
    fn new(history_bits: u32, initial_value: u8) -> Self {
        PerfectBimodalGHistPredictor {
            table: HashMap::new(),
            global_history: 0,
            history_bits,
            initial_value,
        }
    }
}
impl BranchPredictor for PerfectBimodalGHistPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir {
        let history_mask = (1 << self.history_bits) - 1;
        let history = self.global_history & history_mask;
        let counter = self.table.entry((pc, history)).or_insert(self.initial_value);
        let prediction = if *counter >= 2 { BranchDir::Taken } else { BranchDir::NotTaken };

        self.global_history = (self.global_history << 1) | match prediction {
            BranchDir::Taken => 1,
            BranchDir::NotTaken => 0,
        };

        prediction
    }

    fn update(&mut self, pc: u32, inst: BranchInst, actual: BranchDir) {
        let history_mask = (1 << self.history_bits) - 1;
        let history = (self.global_history >> 1) & history_mask; // hack but whatever
        let counter = self.table.entry((pc, history)).or_insert(self.initial_value);
        match actual {
            BranchDir::Taken => {
                if *counter < 3 {
                    *counter += 1;
                }
            }
            BranchDir::NotTaken => {
                if *counter > 0 {
                    *counter -= 1;
                }
            }
        };
    }
}

struct PerfectAlloyedPredictor {
    bhr: HashMap<u32, u32>,
    global_history: u32,
    table: HashMap<(u32, u32, u32), u8>,
    local_history_bits: u32,
    global_history_bits: u32,
}

impl PerfectAlloyedPredictor {
    fn new(local_history_bits: u32, global_history_bits: u32) -> Self {
        PerfectAlloyedPredictor {
            bhr: HashMap::new(),
            global_history: 0,
            table: HashMap::new(),
            local_history_bits,
            global_history_bits,
        }
    }
}

impl BranchPredictor for PerfectAlloyedPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir {
        let local_history_mask = (1 << self.local_history_bits) - 1;
        let global_history_mask = (1 << self.global_history_bits) - 1;

        let mut local_history_entry = self.bhr.entry(pc).or_insert(0);
        let local_history = *local_history_entry & local_history_mask;
        let global_history = self.global_history & global_history_mask;

        let counter = self.table.entry((pc, local_history, global_history)).or_insert(2);
        let prediction = if *counter >= 2 { BranchDir::Taken } else { BranchDir::NotTaken };

        self.global_history = (self.global_history << 1) | match prediction {
            BranchDir::Taken => 1,
            BranchDir::NotTaken => 0,
        };
        *local_history_entry = (*local_history_entry << 1) | match prediction {
            BranchDir::Taken => 1,
            BranchDir::NotTaken => 0,
        };

        prediction
    }

    fn update(&mut self, pc: u32, inst: BranchInst, actual: BranchDir) {
        // let local_history_mask = (1 << self.local_history_bits) - 1;
        // let global_history_mask = (1 << self.global_history_bits) - 1;

        // let local_history = self.bhr.entry(pc).or_insert(0) & local_history_mask;
        // let global_history = self.global_history & global_history_mask;

        // let counter = self.table.entry((pc, *local_history, global_history)).or_insert(2);
        // match actual {
        //     BranchDir::Taken => {
        //         if *counter < 3 {
        //             *counter += 1;
        //         }
        //     }
        //     BranchDir::NotTaken => {
        //         if *counter > 0 {
        //             *counter -= 1;
        //         }
        //     }
        // };
    }
}

struct AlwaysTakenPredictor;
impl BranchPredictor for AlwaysTakenPredictor {
    fn predict(&mut self, _pc: u32, _inst: BranchInst) -> BranchDir {
        BranchDir::Taken
    }
}

struct AlwaysNotTakenPredictor;
impl BranchPredictor for AlwaysNotTakenPredictor {
    fn predict(&mut self, _pc: u32, _inst: BranchInst) -> BranchDir {
        BranchDir::NotTaken
    }
}

struct BackwardsTakenPredictor;
impl BranchPredictor for BackwardsTakenPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir {
        if inst.target < pc {
            BranchDir::Taken
        } else {
            BranchDir::NotTaken
        }
    }
}

struct TournamentPredictor {
    predictors: Vec<Box<dyn BranchPredictor>>,
    predictions: Vec<BranchDir>,
    confidences: HashMap<u32, Vec<u8>>,
}

impl TournamentPredictor {
    fn new(predictors: Vec<Box<dyn BranchPredictor>>) -> Self {
        TournamentPredictor {
            predictors,
            predictions: Vec::new(),
            confidences: HashMap::new(),
        }
    }
}

impl BranchPredictor for TournamentPredictor {
    fn predict(&mut self, pc: u32, inst: BranchInst) -> BranchDir {
        self.predictions.clear();
        for predictor in &mut self.predictors {
            let prediction = predictor.predict(pc, inst);
            self.predictions.push(prediction);
        }
        let confidences = self.confidences.entry(pc).or_insert(vec![1; self.predictors.len()]);
        let mut bestconfidence = 0;
        let mut best  = BranchDir::Taken;
        for (i, prediction) in self.predictions.iter().enumerate() {
            if confidences[i] > bestconfidence {
                bestconfidence = confidences[i];
                best = *prediction;
            }
        }

        best
    }

    fn update(&mut self, pc: u32, inst: BranchInst, actual: BranchDir) {
        let confidences = self.confidences.entry(pc).or_insert(vec![1; self.predictors.len()]);
        for (i, predictor) in self.predictors.iter_mut().enumerate() {
            let prediction = self.predictions[i];
            predictor.update(pc, inst, actual);
            if prediction == actual {
                if confidences[i] < 4 {
                    confidences[i] += 1;
                }
            } else {
                if confidences[i] > 0 {
                    confidences[i] -= 1;
                }
            }
        }
    }
}

fn run_and_get_accuracy(mut predictor: impl BranchPredictor, branches: &[(u32, BranchInst, BranchDir)]) -> f64 {
    let mut correct_predictions = 0;
    let total_predictions = branches.len();


    for &(pc, inst, actual) in branches {
        let prediction = predictor.predict(pc, inst);
        if prediction == actual {
            correct_predictions += 1;
        }
        predictor.update(pc, inst, actual);
    }

    return correct_predictions as f64 / total_predictions as f64;
}

fn main() {
    // traces are in traces folder

    eval_predictors();
    // bruteforce();
}

const STATES: usize = 4;

fn bruteforce() {
    let branches: Vec<(u32, BranchInst, BranchDir)> = read_to_string("branch/aes_c1d4.txt").unwrap()
        .lines()
        .map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let pc = u32::from_str_radix(parts[0], 16).unwrap();
            let bits = u32::from_str_radix(parts[1], 16).unwrap();
            let imm = u32::from_str_radix(parts[2], 16).unwrap();
            let taken = parts[3] == "1";
            let inst = BranchInst { bits, target: pc.wrapping_add(imm) };
            let actual = if taken { BranchDir::Taken } else { BranchDir::NotTaken };
            (pc, inst, actual)
        })
        .collect();

    let mut rng = SmallRng::from_os_rng();

    let mut max_pred = 0u64;
    loop {
        let mut transitions = [[0; 2]; STATES];
        for state in 0..STATES {
            for input in 0..2 {
                transitions[state][input] = rng.random::<u32>() % STATES as u32;
            }
        }
        // let mut taken_states = vec![0u8; STATES];
        // for state in &mut taken_states {
        //     *state = if rng.random_bool(0.5) { 1 } else { 0 };
        // }

        let mut taken = [0u32; STATES];
        let mut nottaken = [0u32; STATES];
        let mut state: u32 = 0;
        for &(_pc, _inst, actual) in &branches {
            if actual == BranchDir::Taken {
                taken[state as usize] += 1;
            } else {
                nottaken[state as usize] += 1;
            }
            state = transitions[state as usize][actual as usize];
        }
        let mut correct_predictions = 0u64;
        let mut taken_states = [0u8; STATES];
        for state in 0..STATES {
            taken_states[state] = if taken[state] >= nottaken[state] { 1 } else { 0 };
            correct_predictions += taken[state].max(nottaken[state]) as u64;
        }
        if correct_predictions > max_pred {
            println!("New max predictions: {} / {}", correct_predictions, branches.len());
            println!("Transitions: {:?}", transitions);
            println!("Taken states: {:?}", taken_states);
        }
        max_pred = max_pred.max(correct_predictions);
    }
}

fn eval_predictors() {
    for file in read_dir("traces").unwrap() {
        let file = file.unwrap();
        let trace_txt = read_to_string(file.path()).unwrap();
        let branches: Vec<(u32, BranchInst, BranchDir)> = trace_txt
            .lines()
            .map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let pc = u32::from_str_radix(parts[0], 16).unwrap();
            let bits = u32::from_str_radix(parts[1], 16).unwrap();
            let imm = u32::from_str_radix(parts[2], 16).unwrap();
            let taken = parts[3] == "1";
            let inst = BranchInst { bits, target: pc.wrapping_add(imm) };
            let actual = if taken { BranchDir::Taken } else { BranchDir::NotTaken };
            (pc, inst, actual)
        })
        .collect();
        println!("FILE {:?}", file.path());

        let always_taken = AlwaysTakenPredictor;
        let accuracy = run_and_get_accuracy(always_taken, &branches);
        println!("always taken: {:.2}%", accuracy * 100.0);

        let always_not_taken = AlwaysNotTakenPredictor;
        let accuracy = run_and_get_accuracy(always_not_taken, &branches);
        println!("always not taken: {:.2}%", accuracy * 100.0);

        let backwards_taken = BackwardsTakenPredictor;
        let accuracy = run_and_get_accuracy(backwards_taken, &branches);
        println!("backwards taken: {:.2}%", accuracy * 100.0);

        for initial in [1, 2] {
            let predictor = PerfectBimodalPredictor::new(initial);
            let accuracy = run_and_get_accuracy(predictor, &branches);
            println!("perfect bimodal {initial}: {:.2}%", accuracy * 100.0);
            
            for history_bits in 1..=20 {
                // let predictor = PerfectGshareXorPredictor::new(history_bits, initial);
                // let accuracy = run_and_get_accuracy(predictor, &branches);
                // println!("perfect gshare {}bit {}: {:.2}%", history_bits, initial, accuracy * 100.0);

                let predictor = PerfectBimodalGHistPredictor::new(history_bits, initial);
                let accuracy = run_and_get_accuracy(predictor, &branches);
                println!("perfect bimodal ghist {}bit {}: {:.2}%", history_bits, initial, accuracy * 100.0);
            }
        }

        /* 
        let predictors: Vec<Box<dyn BranchPredictor>> = vec![
            Box::new(PerfectBimodalPredictor::new(2)),
            Box::new(PerfectGshareXorPredictor::new(2, 2)),
            Box::new(PerfectGshareXorPredictor::new(4, 2)),
            Box::new(PerfectGshareXorPredictor::new(8, 2)),
            Box::new(PerfectBimodalGHistPredictor::new(16, 2)),
            Box::new(PerfectBimodalGHistPredictor::new(32, 2)),
            Box::new(PerfectBimodalGHistPredictor::new(64, 2)),
            Box::new(PerfectBimodalGHistPredictor::new(127, 2)),

        ];
        let tournament_predictor = TournamentPredictor::new(predictors);
        let accuracy = run_and_get_accuracy(tournament_predictor, &branches);
        println!("tournament predictor: {:.2}%", accuracy * 100.0);
        */
        println!();
    }
}
