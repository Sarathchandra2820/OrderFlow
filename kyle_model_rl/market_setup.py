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
        self.beta_star   = noise_std_dev / price_std_dev
        self.lambda_star = 0.5 * price_std_dev / noise_std_dev

    def reset(self):
        self.v_ = np.random.normal(self.base_price, self.price_std_dev)
        self.t_ = 0
        self.u_ = np.random.normal(0, self.noise_std_dev)
        self.Sigma_t = self.price_std_dev**2
        self.phase = 'insider'
        self.x_ = 0
        self.y_ = 0
        self.p_ = self.base_price
        return torch.tensor([
            (self.v_ - self.base_price) / self.price_std_dev, 1.0], dtype=torch.float32)

    def calculate_rewards(self):
        scale = self.price_std_dev * self.noise_std_dev   # keeps rewards O(1)
        r_insider = self.x_ * (self.v_ - self.p_) / scale
        r_market_maker = -r_insider
        return r_insider, r_market_maker

    def step(self, action):
        if self.phase == 'insider':
            self.x_ = action
            self.y_ = self.u_ + self.x_
            self.phase = 'market_maker'

            mm_obs = torch.tensor([
                self.y_ / self.noise_std_dev,                       # normalised order flow
                (self.p_ - self.base_price) / self.price_std_dev    # current price level
            ], dtype=torch.float32)
            return mm_obs, None, False

        else:
            self.p_ = self.p_ + action     # p_t = p_{t-1} + Δp
            lambda_t = self.lambda_star
            var_y = (self.beta_star**2 * self.Sigma_t + self.noise_std_dev**2)
            self.Sigma_t = self.Sigma_t - lambda_t**2 * var_y

            r_insider, r_mm = self.calculate_rewards()
            self.t_ += 1
            done = (self.t_ == self.T)

            if not done:
                self.u_ = np.random.normal(0, self.noise_std_dev)
                self.phase = 'insider'
                insider_obs = torch.tensor([
                    (self.v_ - self.p_) / self.price_std_dev,
                    (self.T - self.t_) / self.T
                ], dtype=torch.float32)
                return insider_obs, (r_insider, r_mm), False
            else:
                return None, (r_insider, r_mm), True


class Agent:
    def __init__(self, env, name):
        if not hasattr(env, 'v_') or not hasattr(env, 'y_'):
            raise ValueError("Environment must be reset before initializing the agent.")
        self.env  = env
        self.name = name

        if name == 'insider':
            # ── Actor ──────────────────────────────────────────────────────
            self.network = torch.nn.Sequential(
                torch.nn.Linear(2, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 2)     # outputs (mu, log_var)
            )
            # ── Critic ─────────────────────────────────────────────────────
            # Separate network; same input shape as actor.
            # Outputs a scalar value estimate V(s).
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(2, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )
            self.optimiser = torch.optim.Adam(
                list(self.network.parameters()) + list(self.critic.parameters()), lr=1e-3)

        elif name == 'market_maker':
            # ── Actor ──────────────────────────────────────────────────────
            # Input: [y_norm, p_prev_norm] at each step
            self.lstm         = torch.nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
            self.output_layer = torch.nn.Linear(64, 2)   # outputs (mu, log_var)
            self.hidden_state = None

            # ── Critic ─────────────────────────────────────────────────────
            # Separate LSTM so actor/critic gradients don't interfere.
            self.critic_lstm         = torch.nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
            self.critic_output_layer = torch.nn.Linear(64, 1)
            self.critic_hidden_state = None

            self.optimiser = torch.optim.Adam(
                list(self.lstm.parameters()) +
                list(self.output_layer.parameters()) +
                list(self.critic_lstm.parameters()) +
                list(self.critic_output_layer.parameters()),
                lr=1e-3)
        else:
            raise ValueError("Agent name must be 'insider' or 'market_maker'")

    def act(self, obs):
        """Returns (action, log_prob, value_estimate)."""
        if self.name == 'insider':
            obs_tensor = obs.clone().detach().float().unsqueeze(0)   # (1, 2)

            # Actor
            out    = self.network(obs_tensor).squeeze()
            mu, log_var = out[0], out[1]
            std    = torch.exp(log_var * 0.5).clamp(min=1e-3)
            dist   = torch.distributions.Normal(mu, std)
            sample = dist.sample()
            log_prob = dist.log_prob(sample)

            # Critic
            value = self.critic(obs_tensor).squeeze()                # scalar

            return sample.item(), log_prob, value

        else:
            obs_tensor = obs.clone().detach().float().unsqueeze(0).unsqueeze(0)  # (1,1,2)

            # Actor
            lstm_out, self.hidden_state = self.lstm(obs_tensor, self.hidden_state)
            out = self.output_layer(lstm_out.squeeze(0)).squeeze()
            mu, log_var = out[0], out[1]
            std    = torch.exp(log_var * 0.5).clamp(min=1e-3, max=1.0)
            dist   = torch.distributions.Normal(mu, std)
            sample = dist.sample()
            log_prob = dist.log_prob(sample)

            # Critic (separate LSTM)
            c_out, self.critic_hidden_state = self.critic_lstm(obs_tensor, self.critic_hidden_state)
            value = self.critic_output_layer(c_out.squeeze(0)).squeeze()   # scalar

            return sample.item(), log_prob, value


if __name__ == "__main__":
    env = KyleMarketEnv(base_price=100, price_std_dev=10, noise_std_dev=5, T=10)
    obs = env.reset()
    insider      = Agent(env, 'insider')
    market_maker = Agent(env, 'market_maker')

    done = False
    while not done:
        if env.phase == 'insider':
            action, log_prob, value = insider.act(obs)
        else:
            action, log_prob, value = market_maker.act(obs)

        obs, rewards, done = env.step(action)
        if rewards is not None:
            r_insider, r_mm = rewards
            print(f"Round {env.t_}: Insider reward: {r_insider:.4f}, MM reward: {r_mm:.4f}")
