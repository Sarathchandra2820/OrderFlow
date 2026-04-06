from market_setup import KyleMarketEnv, Agent
import torch
import os


# ── Discounted returns ─────────────────────────────────────────────────────────
def compute_discounted_rewards(rewards, gamma=1.0):
    G, cumulative = [], 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        G.insert(0, cumulative)
    return G


# ── A2C loss ───────────────────────────────────────────────────────────────────
def compute_a2c_loss(log_probs, rewards, values, critic_coef=0.5):
    """
    log_probs : list of episodes, each a list of scalar tensors
    rewards   : list of episodes, each a list of floats
    values    : list of episodes, each a list of scalar tensors (critic estimates)
    critic_coef: weight on critic (MSE) loss relative to actor loss
    """
    actor_loss  = torch.tensor(0.0)
    critic_loss = torch.tensor(0.0)
    count = 0

    for ep_lp, ep_r, ep_v in zip(log_probs, rewards, values):
        ep_G = compute_discounted_rewards(ep_r)

        for lp, G_t, v in zip(ep_lp, ep_G, ep_v):
            advantage   = G_t - v.detach()          # detach so critic doesn't affect actor grad
            actor_loss  = actor_loss  - lp * advantage
            critic_loss = critic_loss + (G_t - v) ** 2
            count += 1

    count = max(1, count)
    return (actor_loss + critic_coef * critic_loss) / count


# ── Environment & agents ───────────────────────────────────────────────────────
env      = KyleMarketEnv(base_price=100, price_std_dev=10, noise_std_dev=5, T=10)
obs      = env.reset()
insider  = Agent(env, 'insider')
mm_agent = Agent(env, 'market_maker')

num_of_epochs   = 2000
num_of_episodes = 20
log_every       = 10


# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate_trading_behavior(env, insider, mm_agent, num_eval_episodes=3):
    print("\n=== Trading Behavior Check ===")
    print("Columns: t | v_true | insider_x | order_flow_y | mm_price_p | r_insider | r_mm")

    total_insider, total_mm = 0.0, 0.0

    for ep in range(num_eval_episodes):
        obs  = env.reset()
        mm_agent.hidden_state        = None
        mm_agent.critic_hidden_state = None
        ep_i, ep_m = 0.0, 0.0
        done = False

        print(f"\nEpisode {ep+1}/{num_eval_episodes} | true value v={env.v_:.4f}")

        while not done:
            with torch.no_grad():
                x, _, _ = insider.act(obs)
            mm_obs, _, _ = env.step(x)

            with torch.no_grad():
                p, _, _ = mm_agent.act(mm_obs)
            obs, rewards, done = env.step(p)

            r_i, r_m = rewards
            ep_i += r_i; ep_m += r_m
            print(f"  {env.t_:2d} | {env.v_:7.4f} | {x:10.4f} | {env.y_:12.4f} | "
                  f"{env.p_:10.4f} | {r_i:9.4f} | {r_m:8.4f}")

        total_insider += ep_i; total_mm += ep_m
        print(f"  Episode summary -> Insider: {ep_i:.4f}, MM: {ep_m:.4f}")

    n = max(1, num_eval_episodes)
    print(f"\n=== Evaluation Summary ===")
    print(f"Avg episode reward Insider: {total_insider/n:.4f}")
    print(f"Avg episode reward MM:      {total_mm/n:.4f}")


# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(num_of_epochs):

    insider_log_probs, mm_log_probs   = [], []
    insider_rewards,   mm_rewards     = [], []
    insider_values,    mm_values      = [], []

    for episode in range(num_of_episodes):

        obs = env.reset()
        mm_agent.hidden_state        = None
        mm_agent.critic_hidden_state = None

        ep_i_lp, ep_m_lp = [], []
        ep_i_r,  ep_m_r  = [], []
        ep_i_v,  ep_m_v  = [], []

        done = False
        while not done:

            # Insider acts
            x, lp_i, v_i = insider.act(obs)
            mm_obs, _, _  = env.step(x)

            # Market maker acts
            p, lp_m, v_m  = mm_agent.act(mm_obs)
            obs, rewards, done = env.step(p)

            ep_i_lp.append(lp_i);  ep_m_lp.append(lp_m)
            ep_i_r.append(rewards[0]); ep_m_r.append(rewards[1])
            ep_i_v.append(v_i);    ep_m_v.append(v_m)

        insider_log_probs.append(ep_i_lp); mm_log_probs.append(ep_m_lp)
        insider_rewards.append(ep_i_r);    mm_rewards.append(ep_m_r)
        insider_values.append(ep_i_v);     mm_values.append(ep_m_v)

    # Compute A2C losses
    insider_loss = compute_a2c_loss(insider_log_probs, insider_rewards, insider_values)
    mm_loss      = compute_a2c_loss(mm_log_probs,      mm_rewards,      mm_values)

    avg_i = sum(sum(ep) for ep in insider_rewards) / max(1, sum(len(ep) for ep in insider_rewards))
    avg_m = sum(sum(ep) for ep in mm_rewards)      / max(1, sum(len(ep) for ep in mm_rewards))

    # Update insider
    insider.optimiser.zero_grad()
    insider_loss.backward()
    i_grad = torch.nn.utils.clip_grad_norm_(
        list(insider.network.parameters()) + list(insider.critic.parameters()), max_norm=1.0)
    insider.optimiser.step()

    # Update market maker
    mm_agent.optimiser.zero_grad()
    mm_loss.backward()
    m_grad = torch.nn.utils.clip_grad_norm_(
        list(mm_agent.lstm.parameters()) +
        list(mm_agent.output_layer.parameters()) +
        list(mm_agent.critic_lstm.parameters()) +
        list(mm_agent.critic_output_layer.parameters()), max_norm=1.0)
    mm_agent.optimiser.step()

    if epoch < 5 or (epoch + 1) % log_every == 0:
        print(
            f"Epoch {epoch+1:4d}/{num_of_epochs} | "
            f"Insider loss: {insider_loss.item(): .6f} | "
            f"MM loss: {mm_loss.item(): .6f} | "
            f"Avg rewards (I/MM): {avg_i: .4f}/{avg_m: .4f} | "
            f"Grad norms (I/MM): {float(i_grad): .4f}/{float(m_grad): .4f}"
        )

# ── Save models ────────────────────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
torch.save(insider.network.state_dict(),  os.path.join(_dir, 'insider_model.pt'))
torch.save(insider.critic.state_dict(),   os.path.join(_dir, 'insider_critic.pt'))
torch.save({
    'lstm':   mm_agent.lstm.state_dict(),
    'output': mm_agent.output_layer.state_dict(),
    'critic_lstm':   mm_agent.critic_lstm.state_dict(),
    'critic_output': mm_agent.critic_output_layer.state_dict(),
}, os.path.join(_dir, 'mm_model.pt'))
print("Models saved to kyle_model_rl/")

evaluate_trading_behavior(env, insider, mm_agent, num_eval_episodes=3)
