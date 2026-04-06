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
import matplotlib.gridspec as gridspec
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
            x, _, _ = insider.act(obs)
        mm_obs, _, _ = env.step(x)
        with torch.no_grad():
            p, _, _ = mm_agent.act(mm_obs)
        obs, rewards, done = env.step(p)
        r_i, r_mm = rewards
        ep['steps'].append({
            't':          env.t_,
            'x':          x,
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
def ols_line(x, y):
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return slope, intercept, r**2


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Price Discovery (6 representative episodes)
# ══════════════════════════════════════════════════════════════════════════════
# Sort episodes by |v - base_price| so we show diverse cases
sorted_eps = sorted(episodes, key=lambda ep: abs(ep['v'] - env.base_price), reverse=True)
show_eps   = sorted_eps[:3] + sorted_eps[N_EVAL//2 - 1: N_EVAL//2 + 2]  # high + mid mispricing

fig1, axes = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
fig1.suptitle("Figure 1 – Price Discovery Across Episodes", fontsize=14, fontweight='bold')

T = env.T
steps_range = range(1, T + 1)

for col, ep in enumerate(show_eps):
    v = ep['v']
    prices = [s['p'] for s in ep['steps']]
    trades = [s['x'] for s in ep['steps']]
    rewards = [s['r_insider'] for s in ep['steps']]
    cum_rewards = np.cumsum(rewards)
    ts = [s['t'] for s in ep['steps']]

    # ── Row 0: price path ──────────────────────────────────────────────────
    ax = axes[0, col]
    ax.axhline(v,             color=STYLE['true_val'], linestyle='--', lw=1.5, label='True value v')
    ax.axhline(env.base_price, color=STYLE['neutral'],  linestyle=':',  lw=1.0, label='Prior μ₀')
    ax.plot(ts, prices, 'o-', color=STYLE['price'], lw=2, ms=5, label='MM price p_t')
    ax.fill_between(ts, env.base_price, prices, alpha=0.12, color=STYLE['price'])
    ax.set_title(f"v = {v:.1f}  (Δ = {v - env.base_price:+.1f})", fontsize=9)
    ax.set_ylabel("Price" if col == 0 else "")
    ax.set_xlim(0.5, T + 0.5)
    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.3)

    # ── Row 1: insider trade size ──────────────────────────────────────────
    ax = axes[1, col]
    colors = [STYLE['trade'] if xi > 0 else STYLE['true_val'] for xi in trades]
    ax.bar(ts, trades, color=colors, alpha=0.75, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    # theoretical beta* × mispricing at t=1
    misprice_0 = v - env.base_price
    ax.axhline(env.beta_star * misprice_0, color=STYLE['theory'],
               linestyle='--', lw=1.2, label=f'β*·(v−μ₀)={env.beta_star * misprice_0:.1f}')
    ax.set_ylabel("Trade x" if col == 0 else "")
    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.3, axis='y')
    if col == 0:
        ax.legend(fontsize=7)

    # ── Row 2: cumulative insider reward ───────────────────────────────────
    ax = axes[2, col]
    ax.plot(ts, cum_rewards, 's-', color=STYLE['trade'], lw=2, ms=5)
    ax.fill_between(ts, 0, cum_rewards, alpha=0.15, color=STYLE['trade'])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumul. reward" if col == 0 else "")
    ax.grid(True, alpha=0.3)

# shared legend for row 0
handles = [
    Line2D([0], [0], color=STYLE['true_val'], linestyle='--', lw=1.5, label='True value v'),
    Line2D([0], [0], color=STYLE['neutral'],  linestyle=':',  lw=1.0, label='Prior μ₀=100'),
    Line2D([0], [0], color=STYLE['price'],    lw=2,          label='MM price p_t'),
]
fig1.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, -0.01))

fig1.savefig(os.path.join(MODEL_DIR, 'fig1_price_discovery.png'), dpi=150, bbox_inches='tight')
print("Saved fig1_price_discovery.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Agent Strategy Scatter Plots
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig2.suptitle("Figure 2 – Learned vs Theoretical Strategies", fontsize=14, fontweight='bold')

# ── Left: Insider strategy  x vs (v - p_prev) ─────────────────────────────
ax = axes[0]
ax.scatter(all_mispricing, all_x, alpha=0.25, s=18, color=STYLE['price'], label='Observations')

# OLS fit
slope_i, intercept_i, r2_i = ols_line(all_mispricing, all_x)
xr = np.linspace(all_mispricing.min(), all_mispricing.max(), 200)
ax.plot(xr, slope_i * xr + intercept_i, color=STYLE['price'], lw=2,
        label=f'OLS fit  β={slope_i:.3f}, R²={r2_i:.2f}')

# Theoretical: x = beta_star * mispricing (through origin, no intercept)
ax.plot(xr, env.beta_star * xr, color=STYLE['theory'], lw=2, linestyle='--',
        label=f'Theory β*={env.beta_star}  (through origin)')

ax.axhline(0, color='black', lw=0.7)
ax.axvline(0, color='black', lw=0.7)
ax.set_xlabel("Mispricing  v − p_{t−1}", fontsize=11)
ax.set_ylabel("Insider trade  x_t", fontsize=11)
ax.set_title("Insider Strategy", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Right: MM strategy  Δp vs y (order flow) ──────────────────────────────
ax = axes[1]
ax.scatter(all_y, all_delta_p, alpha=0.25, s=18, color=STYLE['true_val'], label='Observations')

slope_m, intercept_m, r2_m = ols_line(all_y, all_delta_p)
yr = np.linspace(all_y.min(), all_y.max(), 200)
ax.plot(yr, slope_m * yr + intercept_m, color=STYLE['true_val'], lw=2,
        label=f'OLS fit  λ={slope_m:.3f}, R²={r2_m:.2f}')

# Theoretical: Δp = lambda_star * y (through origin)
ax.plot(yr, env.lambda_star * yr, color=STYLE['theory'], lw=2, linestyle='--',
        label=f'Theory λ*={env.lambda_star}  (through origin)')

ax.axhline(0, color='black', lw=0.7)
ax.axvline(0, color='black', lw=0.7)
ax.set_xlabel("Order flow  y_t = x_t + u_t", fontsize=11)
ax.set_ylabel("Price update  Δp_t", fontsize=11)
ax.set_title("Market Maker Strategy", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig2.savefig(os.path.join(MODEL_DIR, 'fig2_strategies.png'), dpi=150, bbox_inches='tight')
print("Saved fig2_strategies.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Performance Summary
# ══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(14, 10), constrained_layout=True)
fig3.suptitle("Figure 3 – Performance Summary", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig3)

# ── Top-left: terminal price vs true value ─────────────────────────────────
ax = fig3.add_subplot(gs[0, 0])
ax.scatter(all_v, all_p_final, alpha=0.6, s=40, color=STYLE['price'], zorder=3)
lims = [min(all_v.min(), all_p_final.min()) - 2, max(all_v.max(), all_p_final.max()) + 2]
ax.plot(lims, lims, 'k--', lw=1.2, label='Perfect pricing (p=v)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True value  v", fontsize=10)
ax.set_ylabel("Terminal MM price  p_T", fontsize=10)
ax.set_title("Terminal Price vs True Value", fontsize=11, fontweight='bold')
rmse = np.sqrt(np.mean((all_p_final - all_v)**2))
ax.text(0.05, 0.92, f'RMSE = {rmse:.2f}', transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Top-right: per-round reward distribution (box plot) ───────────────────
ax = fig3.add_subplot(gs[0, 1])
rewards_by_t = [
    [ep['steps'][t]['r_insider'] for ep in episodes if len(ep['steps']) > t]
    for t in range(T)
]
bp = ax.boxplot(rewards_by_t, positions=range(1, T + 1), patch_artist=True,
                medianprops=dict(color='black', lw=2))
for patch in bp['boxes']:
    patch.set_facecolor(STYLE['price'])
    patch.set_alpha(0.5)
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.set_xlabel("Round t", fontsize=10)
ax.set_ylabel("Insider reward  r_t", fontsize=10)
ax.set_title("Per-Round Insider Reward Distribution", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# ── Bottom-left: terminal mispricing histogram ─────────────────────────────
ax = fig3.add_subplot(gs[1, 0])
terminal_misprice = all_v - all_p_final
ax.hist(terminal_misprice, bins=20, color=STYLE['true_val'], alpha=0.7, edgecolor='white')
ax.axvline(0,                         color='black',        lw=1.5, linestyle='--', label='Zero residual')
ax.axvline(terminal_misprice.mean(),  color=STYLE['theory'], lw=2,   linestyle='-',
           label=f'Mean = {terminal_misprice.mean():.2f}')
ax.set_xlabel("Residual mispricing  v − p_T", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Terminal Mispricing  (v − p_T)", fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# ── Bottom-right: cumulative reward per episode ────────────────────────────
ax = fig3.add_subplot(gs[1, 1])
ep_total_rewards = [sum(s['r_insider'] for s in ep['steps']) for ep in episodes]
colors_ep = [STYLE['trade'] if r > 0 else STYLE['true_val'] for r in ep_total_rewards]
ax.bar(range(1, N_EVAL + 1), ep_total_rewards, color=colors_ep, alpha=0.75)
ax.axhline(np.mean(ep_total_rewards), color='black', lw=1.5, linestyle='--',
           label=f'Mean = {np.mean(ep_total_rewards):.1f}')
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel("Episode", fontsize=10)
ax.set_ylabel("Total insider profit", fontsize=10)
ax.set_title("Total Insider Profit per Episode", fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

fig3.savefig(os.path.join(MODEL_DIR, 'fig3_performance.png'), dpi=150, bbox_inches='tight')
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
