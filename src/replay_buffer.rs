use rand::{rngs::StdRng, Rng};
use tch::{Tensor, kind::{FLOAT_CUDA, INT64_CUDA}};

pub struct ReplayBuffer {
    pub state: Tensor,
    pub next_state: Tensor,
    pub rewards: Tensor,
    pub actions: Tensor,
    pub terminated: Tensor,
    pub timedout: Tensor,
    pub capacity: usize,
    pub len: usize,
    pub i: usize
}

impl ReplayBuffer {
    pub fn new(
        capacity: usize, 
        num_obs: usize, 
        _num_actions: usize
    ) -> Self {
        Self {
            state: Tensor::zeros([capacity as _, num_obs as _], FLOAT_CUDA),
            next_state: Tensor::zeros([capacity as _, num_obs as _], FLOAT_CUDA),
            rewards: Tensor::zeros([capacity as _, 1], FLOAT_CUDA),
            actions: Tensor::zeros([capacity as _, 1], INT64_CUDA),
            terminated: Tensor::zeros([capacity as _, 1], INT64_CUDA),
            timedout: Tensor::zeros([capacity as _, 1], INT64_CUDA),
            capacity,
            len: 0,
            i: 0
        }
    }

    pub fn store(
        &mut self, 
        state: &Tensor, 
        actions: &Tensor, 
        reward: &Tensor, 
        next_state: &Tensor,
        terminated: &Tensor,
        timedout: &Tensor
    ) {
        let i = self.i % self.capacity;
        self.state.get(i as _).copy_(state);
        self.rewards.get(i as _).copy_(reward);
        self.actions.get(i as _).copy_(actions);
        self.next_state.get(i as _).copy_(next_state);
        self.terminated.get(i as _).copy_(terminated);
        self.timedout.get(i as _).copy_(timedout);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn sample(
        &mut self, 
        batch_size: usize,
        rng: &mut StdRng
    ) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = vec![rng.gen_range(0..self.len-2) as i64; batch_size];
        let batch_indexes_ts = Tensor::from_slice(&batch_indexes).to_device(tch::Device::Cuda(0));

        let states = self.state.index_select(0, &batch_indexes_ts);
        let next_states = self.next_state.index_select(0, &batch_indexes_ts);
        let actions = self.actions.index_select(0, &batch_indexes_ts).flatten(0, 1);
        let rewards = self.rewards.index_select(0, &batch_indexes_ts);
        let terminated = self.terminated.index_select(0, &batch_indexes_ts);
        let timedout = self.timedout.index_select(0, &batch_indexes_ts);

        Some((states, actions, rewards, next_states, terminated, timedout))
    }
}