use rand::rngs::StdRng;
use tch::Tensor;

use crate::baseagent::BaseAgentOptions;

pub trait Agent {
    fn new(
        options: &BaseAgentOptions
    ) -> Self where Self: Sized;

    fn choose_action(
        &mut self, 
        obs: &Tensor,
        rng: &mut StdRng
    ) -> Option<Tensor>;

    fn store_transition(
        &mut self, 
        state: &Tensor, 
        action: &Tensor, 
        reward: &Tensor, 
        next_state: &Tensor,
        terminated: &Tensor,
        timedout: &Tensor
    );

    fn learn(
        &mut self,
        rng: &mut StdRng
    );

    fn save(
        &self,
        path: String
    ) -> Result<(), String>;

    fn load(
        &mut self,
        path: String
    ) -> Result<(), String>;

    fn get_epsilon(
        &self
    ) -> f64;
}