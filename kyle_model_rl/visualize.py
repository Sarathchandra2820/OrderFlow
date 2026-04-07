"""
Kyle Model RL – Visualisation
Run from project root:  python kyle_model_rl/visualize.py
Requires insider_model.pt and mm_model.pt saved by simulation.py.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from market_setup import KyleMarketEnv, Agent

# ── Config ─────────────────────────────────────────────────────────────────────
N_EVAL   = 50
SEED     = 0
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(SEED)
np.random.seed(SEED)

STYLE = {
    'price':    '#2563EB',   # blue  – MM price path
    'true_val': '#DC2626',   # red   – true asset value
    'trade':    '#16A34A',   # green – insider trade
    'theory':   '#F97316',   # orange – theoretical benchmark
    'neutral':  '#6B7280',   # grey
}

# ── Load environment and agents ────────────────────────────────────────────────
env = KyleMarketEnv(base_price=100, price_std_dev=10, noise_std_dev=5, T=10)
obs = env.reset()
insider  = Agent(env, 'insider')
mm_agent = Agent(env, 'market_maker')

insider.network.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, 'insider_model.pt'), map_location='cpu'))
insider.critic.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, 'insider_critic.pt'), map_location='cpu'))
ckpt = torch.load(os.path.join(MODEL_DIR, 'mm_model.pt'), map_location='cpu')
mm_agent.lstm.load_state_dict(ckpt['lstm'])
mm_agent.output_layer.load_state_dict(ckpt['output'])
mm_agent.critic_lstm.load_state_dict(ckpt['critic_lstm'])
mm_agent.critic_output_layer.load_state_dict(ckpt['critic_output'])

insider.network.eval(); insider.critic.eval()
mm_agent.lstm.eval();   mm_agent.output_layer.eval()
mm_agent.critic_lstm.eval(); mm_agent.critic_output_layer.eval()

# ── Collect evaluation episodes ────────────────────────────────────────────────
episodes = []
for _ in range(N_EVAL):
    obs = env.reset()
    mm_agent.hidden_state        = None
    mm_agent.critic_hidden_state = None
    v = env.v_
    p_prev = env.base_price
    ep = {'v': v, 'steps': []}
    done = False
    while not done:
        with torch.no_grad():
            insider_trade, _, _ = insider.act(obs)
        mm_obs, _, _ = env.step(insider_trade)
        with torch.no_grad():
            p, _, _ = mm_agent.act(mm_obs)
        obs, rewards, done = env.step(p)
        r_i, r_mm = rewards
        ep['steps'].append({
            't':          env.t_,
            'x':          insider_trade,
            'y':          env.y_,
            'p':          env.p_,
            'delta_p':    env.p_ - p_prev,
            'r_insider':  r_i,
            'mispricing': v - p_prev,   # mispricing BEFORE this MM price update
        })
        p_prev = env.p_
    episodes.append(ep)

# ── Flatten arrays ─────────────────────────────────────────────────────────────
all_x          = np.array([s['x']          for ep in episodes for s in ep['steps']])
all_y          = np.array([s['y']          for ep in episodes for s in ep['steps']])
all_delta_p    = np.array([s['delta_p']    for ep in episodes for s in ep['steps']])
all_mispricing = np.array([s['mispricing'] for ep in episodes for s in ep['steps']])
all_r_insider  = np.array([s['r_insider']  for ep in episodes for s in ep['steps']])
all_v          = np.array([ep['v']         for ep in episodes])
all_p_final    = np.array([ep['steps'][-1]['p'] for ep in episodes])

# ── Helper: OLS line ───────────────────────────────────────────────────────────
def ols_line(x_vals, y_vals):
    slope, intercept, r, _, _ = stats.linregress(x_vals, y_vals)
    return slope, intercept, r**2


def pick_representative_episodes(eps, k=4):
    scored = sorted(eps, key=lambda ep: abs(ep['v'] - env.base_price))
    if len(scored) <= k:
        return scored
    idx = np.linspace(0, len(scored) - 1, k).astype(int)
    return [scored[i] for i in idx]


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titleweight': 'bold',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.frameon': False,
    'grid.alpha': 0.2,
    'figure.dpi': 110,
})


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Representative Price Paths (streamlined)
# ══════════════════════════════════════════════════════════════════════════════
T = env.T
show_eps = pick_representative_episodes(episodes, k=4)

fig1, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
fig1.suptitle("Figure 1 – Representative Price Discovery", fontsize=14, fontweight='bold')

for ax, ep in zip(axes.flat, show_eps):
    v = ep['v']
    ts = [s['t'] for s in ep['steps']]
    prices = [s['p'] for s in ep['steps']]

    ax.axhline(v, color=STYLE['true_val'], linestyle='--', lw=1.5)
    ax.axhline(env.base_price, color=STYLE['neutral'], linestyle=':', lw=1.1)
    ax.plot(ts, prices, marker='o', lw=2.2, ms=4.5, color=STYLE['price'])
    ax.fill_between(ts, env.base_price, prices, alpha=0.10, color=STYLE['price'])

    ax.set_title(f"v={v:.1f} (Δ={v - env.base_price:+.1f})")
    ax.set_xlim(1, T)
    ax.set_xticks(range(1, T + 1, 2))

for row in axes:
    row[0].set_ylabel("Price")
for ax in axes[-1]:
    ax.set_xlabel("Round t")

handles = [
    Line2D([0], [0], color=STYLE['price'], lw=2.2, label='MM price $p_t$'),
    Line2D([0], [0], color=STYLE['true_val'], linestyle='--', lw=1.5, label='True value $v$'),
    Line2D([0], [0], color=STYLE['neutral'], linestyle=':', lw=1.1, label='Prior μ₀'),
]
fig1.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9)
fig1.savefig(os.path.join(MODEL_DIR, 'fig1_price_discovery.png'), dpi=160, bbox_inches='tight')
print("Saved fig1_price_discovery.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Learned Strategies (denser, cleaner)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
fig2.suptitle("Figure 2 – Learned vs Theoretical Strategies", fontsize=14, fontweight='bold')

# Insider strategy: x vs mispricing
ax = axes[0]
hb = ax.hexbin(all_mispricing, all_x, gridsize=28, cmap='Blues', mincnt=1)
slope_i, intercept_i, r2_i = ols_line(all_mispricing, all_x)
xr = np.linspace(all_mispricing.min(), all_mispricing.max(), 200)
ax.plot(xr, slope_i * xr + intercept_i, color=STYLE['price'], lw=2.2,
        label=f'Fit: β={slope_i:.3f}, R²={r2_i:.2f}')
ax.plot(xr, env.beta_star * xr, color=STYLE['theory'], lw=2.0, linestyle='--',
        label=f'Theory: β*={env.beta_star}')
ax.axhline(0, color='black', lw=0.7)
ax.axvline(0, color='black', lw=0.7)
ax.set_xlabel("Mispricing $v - p_{t-1}$")
ax.set_ylabel("Insider trade $x_t$")
ax.set_title("Insider Strategy")
ax.legend(fontsize=9)
fig2.colorbar(hb, ax=ax, label='Count')

# MM strategy: Δp vs order flow
ax = axes[1]
hb = ax.hexbin(all_y, all_delta_p, gridsize=28, cmap='Reds', mincnt=1)
slope_m, intercept_m, r2_m = ols_line(all_y, all_delta_p)
yr = np.linspace(all_y.min(), all_y.max(), 200)
ax.plot(yr, slope_m * yr + intercept_m, color=STYLE['true_val'], lw=2.2,
        label=f'Fit: λ={slope_m:.3f}, R²={r2_m:.2f}')
ax.plot(yr, env.lambda_star * yr, color=STYLE['theory'], lw=2.0, linestyle='--',
        label=f'Theory: λ*={env.lambda_star}')
ax.axhline(0, color='black', lw=0.7)
ax.axvline(0, color='black', lw=0.7)
ax.set_xlabel("Order flow $y_t = x_t + u_t$")
ax.set_ylabel("Price update Δp_t")
ax.set_title("Market Maker Strategy")
ax.legend(fontsize=9)
fig2.colorbar(hb, ax=ax, label='Count')

fig2.savefig(os.path.join(MODEL_DIR, 'fig2_strategies.png'), dpi=160, bbox_inches='tight')
print("Saved fig2_strategies.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Performance Dashboard (compact)
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig3.suptitle("Figure 3 – Performance Summary", fontsize=14, fontweight='bold')

# Top-left: terminal price vs true value
ax = axes[0, 0]
ax.scatter(all_v, all_p_final, alpha=0.65, s=34, color=STYLE['price'])
lims = [min(all_v.min(), all_p_final.min()) - 2, max(all_v.max(), all_p_final.max()) + 2]
ax.plot(lims, lims, 'k--', lw=1.2)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("True value $v$")
ax.set_ylabel("Terminal price $p_T$")
ax.set_title("Terminal Price Accuracy")
rmse = np.sqrt(np.mean((all_p_final - all_v) ** 2))
ax.text(0.04, 0.93, f'RMSE={rmse:.2f}', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontsize=9)

# Top-right: terminal mispricing histogram
ax = axes[0, 1]
terminal_misprice = all_v - all_p_final
ax.hist(terminal_misprice, bins=16, color=STYLE['true_val'], alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', lw=1.2, linestyle='--')
ax.axvline(terminal_misprice.mean(), color=STYLE['theory'], lw=2.0)
ax.set_xlabel("Residual $v - p_T$")
ax.set_ylabel("Count")
ax.set_title("Terminal Mispricing")

# Bottom-left: per-round reward distribution
ax = axes[1, 0]
rewards_by_t = [[ep['steps'][t]['r_insider'] for ep in episodes if len(ep['steps']) > t] for t in range(T)]
bp = ax.boxplot(rewards_by_t, positions=range(1, T + 1), widths=0.65, patch_artist=True,
                medianprops=dict(color='black', lw=1.8), showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor(STYLE['price'])
    patch.set_alpha(0.45)
ax.axhline(0, color='black', lw=1.0, linestyle='--')
ax.set_xlabel("Round t")
ax.set_ylabel("Reward $r_t$")
ax.set_title("Per-Round Insider Reward")

# Bottom-right: total reward per episode (sorted)
ax = axes[1, 1]
ep_total_rewards = np.array([sum(s['r_insider'] for s in ep['steps']) for ep in episodes])
sorted_rewards = np.sort(ep_total_rewards)
ax.plot(np.arange(1, len(sorted_rewards) + 1), sorted_rewards, color=STYLE['trade'], lw=2)
ax.fill_between(np.arange(1, len(sorted_rewards) + 1), 0, sorted_rewards,
                color=STYLE['trade'], alpha=0.12)
ax.axhline(sorted_rewards.mean(), color='black', lw=1.2, linestyle='--')
ax.axhline(0, color='black', lw=1.0)
ax.set_xlabel("Episode rank (by profit)")
ax.set_ylabel("Total insider profit")
ax.set_title("Episode Profit Distribution")

fig3.savefig(os.path.join(MODEL_DIR, 'fig3_performance.png'), dpi=160, bbox_inches='tight')
print("Saved fig3_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Episodes evaluated      : {N_EVAL}")
print(f"  Avg insider profit/ep   : {np.mean(ep_total_rewards):.2f}")
print(f"  Avg insider profit/step : {all_r_insider.mean():.2f}")
print(f"  Terminal price RMSE     : {rmse:.2f}")
print(f"  Residual mispricing μ   : {terminal_misprice.mean():.2f}")
print(f"  Insider strategy  β_fit : {slope_i:.3f}  (theory β*={env.beta_star})")
print(f"  MM strategy       λ_fit : {slope_m:.3f}  (theory λ*={env.lambda_star})")
print(f"  Insider R²              : {r2_i:.3f}")
print(f"  MM R²                   : {r2_m:.3f}")
print("="*60)

plt.show()
