import numpy as np
from dataclasses import dataclass
import torch
import random 


class KyleMarketEnv:
    def __init__(self, base_price, price_std_dev, noise_std_dev, T):
        self.base_price = base_price
        self.price_std_dev = price_std_dev
        self.noise_std_dev  = noise_std_dev
        self.T = T
        # analytical benchmark — useful for diagnostics later
        self.beta_star = noise_std_dev / price_std_dev
        self.lambda_star = 0.5 * price_std_dev / noise_std_dev  # correct (lambda_star is fine)

    def reset(self):
        self.v_ = np.random.normal(self.base_price, self.price_std_dev)
        self.t_ = 0
        self.u_ = np.random.normal(0, self.noise_std_dev)
        self.Sigma_t = self.price_std_dev**2
        self.phase = 'insider'
        self.x_ = 0
        self.y_ = 0
        self.p_ = self.base_price  # fix discussed below
        return torch.tensor([
            (self.v_ - self.base_price) / self.price_std_dev, 1.0], dtype=torch.float32)

    
    def calculate_rewards(self):
        r_insider = self.x_ * (self.v_ - self.p_)
        r_market_maker = -r_insider
        return r_insider, r_market_maker

    
    def step(self, action):
        '''
        This is a branch logic for the two phases of the game. The insider first chooses how many shares to buy/sell, then the market maker sets the price. The reward is calculated based on the difference between the true value and the price set by the market maker, multiplied by the number of shares bought/sold by the insider.
        '''
        if self.phase == 'insider':
            self.x_ = action
            self.y_ = self.u_ + self.x_
            self.phase = 'market_maker'

            mm_obs = torch.tensor([self.y_ / self.noise_std_dev], dtype=torch.float32)  # market maker observes normalized order flow
            return mm_obs, None, False  # Market maker observes y

        else:
            self.p_ = action               # market maker updates price
            # update analytical Sigma_t for diagnostics
            lambda_t = self.lambda_star    # or compute from current Sigma_t
            var_y = (self.beta_star**2 * self.Sigma_t + self.noise_std_dev**2)
            self.Sigma_t = self.Sigma_t - lambda_t**2 * var_y
            
            r_insider, r_mm = self.calculate_rewards()
            
            self.t_ += 1
            done = (self.t_ == self.T)
            
            if not done:
                self.u_ = np.random.normal(0, self.noise_std_dev)
                self.phase = 'insider'
                # insider's observation for next round: updated mispricing after MM moved price
                insider_obs = torch.tensor([
                    (self.v_ - self.p_) / self.price_std_dev,  # mispricing has shrunk
                    (self.T - self.t_) / self.T                 # time remaining
                ], dtype=torch.float32)
                return insider_obs, (r_insider, r_mm), False
            else:
                return None, (r_insider, r_mm), True




class Agent:
    def __init__(self, env, name):
        #make sure that env is reset before initializing the agent, otherwise the agent will not have access to the correct information
        if not hasattr(env, 'v_') or not hasattr(env, 'y_'):
            raise ValueError("Environment must be reset before initializing the agent.")
        self.env = env
        self.name = name
        self.action = 0
        self.r_agent = 0


        if name == 'insider':
            # feedforward: input is (mispricing, time_remaining) → 2 inputs
            self.network = torch.nn.Sequential(
                torch.nn.Linear(2, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 2)
            )
            #self.sigma = 1.0  # exploration std for Gaussian policy
        elif name == 'market_maker':
            # LSTM: input is y_t at each round → 1 input per step (normalized order flow)
            self.lstm = torch.nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
            self.output_layer = torch.nn.Linear(64, 2)
            self.hidden_state = None  # reset this at the start of each episode
            #self.sigma = 1.0


        if name == 'insider':
            self.info = {'v': env.v_, 'base_price': env.base_price}  # Insider observes v - p
        else:
            self.info = {'y': env.y_}  # Market maker observes y

    def act(self, obs):
        # Placeholder for action logic
        if self.name == 'insider':
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape (1, 2)
            action = self.network(obs_tensor).squeeze()  # shape (1, 2) → (2,)
            mu, log_var = action[0], action[1]
            var = torch.exp(log_var)
            action_dist = torch.distributions.Normal(mu, var)
            sample_action = action_dist.sample()
            log_probs = action_dist.log_prob((sample_action))
            return sample_action.item(), log_probs
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 2)
            lstm_out, self.hidden_state = self.lstm(obs_tensor, self.hidden_state)  # lstm_out shape (1, 1, 64)
            action = self.output_layer(lstm_out.squeeze(0)).squeeze()  # shape (1, 2) → (2,)
            mu, log_var = action[0], action[1]
            var = torch.exp(log_var)
            action_dist = torch.distributions.Normal(mu, var)
            sample_action = action_dist.sample()
            log_probs = action_dist.log_prob((sample_action))
            return sample_action.item(), log_probs
    
    def update_info(self, env):
        if self.name == 'insider':
            self.info = {'v': env.v_, 'current price' : env.p_}  # Insider observes v - p
        # Placeholder for information retrieval logic
        if self.name == 'market_maker':
            self.info = {'y': env.y_}  # Market maker observes y

 
    
    


    
    