"""
Kyle Model RL – Multi-Period Convergence Diagnostics
Run:  python kyle_model_rl/diagnostics.py

Correct multi-period Kyle (1985) equilibrium (T-period discrete model):
  - λ_t = λ* = σ_v/(2σ_u)  CONSTANT across all periods (Kyle 1985 key result)
  - β_t is TIME-VARYING (increases toward end):
        β_t = σ_u * √T / (√Σ_0 * √((T-t)(T-t-1)))    for t = 1 … T-1
        β_T = σ_u * √T / √Σ_0                          (last period, single-period formula)
  - Σ_t = Σ_0 * (T-t)/T   (variance decreases linearly to zero — full price discovery at T)

Derivation of β_t:
  From the Kalman update Σ_{t+1} = Σ_t * σ_u² / (β_t² Σ_t + σ_u²)
  and the linear Σ equilibrium Σ_t = Σ_0*(T-t)/T, solving for β_t gives the above.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from scipy import stats

from market_setup import KyleMarketEnv, Agent

# ── Config ─────────────────────────────────────────────────────────────────────
N_EVAL    = 500
SEED      = 42
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load env & agents ──────────────────────────────────────────────────────────
env = KyleMarketEnv(base_price=100, price_std_dev=10, noise_std_dev=5, T=1)
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

insider.network.eval();  insider.critic.eval()
mm_agent.lstm.eval();    mm_agent.output_layer.eval()
mm_agent.critic_lstm.eval();  mm_agent.critic_output_layer.eval()

T       = env.T
Sigma_0 = env.price_std_dev ** 2   # 100
sigma_u = env.noise_std_dev         # 5
LAMBDA_STAR = env.lambda_star       # 1.0  (constant across all periods)


# ── Correct multi-period theoretical benchmarks ────────────────────────────────
def multi_period_benchmarks(T, Sigma_0, sigma_u):
    """
    Returns arrays (beta_star_t, lambda_star_t, Sigma_t_theory) of length T.
    t is 0-indexed: t=0 is the FIRST trading round, t=T-1 is the LAST.

    Derivation (discrete Kyle, T rounds, competitive MM):
      Equilibrium: Sigma_t = Sigma_0 * (T - t) / T   (linear variance decrease)
      Lambda_t    = lambda* = sigma_v / (2*sigma_u)   (constant)
      Beta_t      = sigma_u * sqrt(T) / (sqrt(Sigma_0) * sqrt((T-t)*(T-t-1)))
                    for t in 0 … T-2
      Beta_{T-1}  = sigma_u * sqrt(T) / sqrt(Sigma_0)   (last period)
    """
    beta_t   = np.zeros(T)
    sigma_t  = np.zeros(T)   # Sigma entering period t (before trading)
    lambda_t = np.full(T, sigma_u / (env.price_std_dev / 2))   # σ_v / (2σ_u)

    for t in range(T):
        sigma_t[t] = Sigma_0 * (T - t) / T    # variance BEFORE period t+1 trades
        remaining  = T - t                     # rounds left including this one

        if remaining > 1:
            # General formula derived from Kalman-update + linear-Sigma condition
            beta_t[t] = sigma_u * np.sqrt(T) / (np.sqrt(Sigma_0) * np.sqrt(remaining * (remaining - 1)))
        else:
            # Last period: single-period Kyle with current conditional variance
            beta_t[t] = sigma_u / np.sqrt(sigma_t[t])

    return beta_t, lambda_t, sigma_t


beta_theory, lambda_theory, sigma_theory = multi_period_benchmarks(T, Sigma_0, sigma_u)

print("\n" + "="*72)
print("  MULTI-PERIOD KYLE (1985) EQUILIBRIUM BENCHMARKS")
print(f"  T={T}, σ_v={env.price_std_dev}, σ_u={sigma_u}, Σ_0={Sigma_0}")
print(f"  λ* = {LAMBDA_STAR:.4f}  (constant across all periods)")
print("="*72)
print(f"  {'t':>3}  {'β_t*':>8}  {'Σ_t (entering)':>16}  {'√Σ_t':>8}")
print("  " + "-"*45)
for t in range(T):
    print(f"  {t+1:>3}  {beta_theory[t]:>8.4f}  {sigma_theory[t]:>16.4f}  {np.sqrt(sigma_theory[t]):>8.4f}")

# ── Collect evaluation episodes ────────────────────────────────────────────────
per_step = [[] for _ in range(T)]

for _ in range(N_EVAL):
    obs = env.reset()
    mm_agent.hidden_state        = None
    mm_agent.critic_hidden_state = None
    v      = env.v_
    p_prev = env.base_price
    done   = False

    while not done:
        with torch.no_grad():
            x, _, _ = insider.act(obs)
        mm_obs, _, _ = env.step(x)
        with torch.no_grad():
            delta_p, _, _ = mm_agent.act(mm_obs)
        obs, rewards, done = env.step(delta_p)

        t_idx = env.t_ - 1
        per_step[t_idx].append({
            'x':          x,
            'y':          env.y_,
            'delta_p':    env.p_ - p_prev,
            'mispricing': v - p_prev,
            'p':          env.p_,
            'v':          v,
            'r_insider':  rewards[0],
            'r_mm':       rewards[1],
        })
        p_prev = env.p_


# ── 1. Per-period β: compare to time-varying theory ───────────────────────────
print("\n" + "="*72)
print("  DIAGNOSTIC 1 – Per-Period Insider Strategy β_t")
print("  Benchmark: multi-period β_t* (increases from 0.17 → 1.58)")
print("  NOT the flat single-period β* = 0.50 (wrong benchmark for T>1)")
print("="*72)
print(f"  {'t':>3}  {'β_t*':>8}  {'β_fit':>8}  {'err%':>8}  {'R²':>7}  {'verdict':>10}")
print("  " + "-"*58)

beta_fits  = []
r2_ins_all = []

for t in range(T):
    d    = per_step[t]
    misp = np.array([s['mispricing'] for s in d])
    x_a  = np.array([s['x']         for s in d])

    b, _, r, _, _ = stats.linregress(misp, x_a)
    r2 = r**2
    beta_fits.append(b)
    r2_ins_all.append(r2)

    err = 100 * (b - beta_theory[t]) / beta_theory[t]
    verdict = "PASS" if abs(err) < 25 else "FAIL"
    print(f"  {t+1:>3}  {beta_theory[t]:>8.4f}  {b:>8.4f}  {err:>+7.1f}%  {r2:>7.3f}  {verdict:>10}")

beta_fits = np.array(beta_fits)
print(f"\n  Key check: does β_fit INCREASE across periods? (should mimic theory)")
diffs_fit    = np.diff(beta_fits)
diffs_theory = np.diff(beta_theory)
monotone_fit = np.sum(diffs_fit > 0)
print(f"  Periods where β increases: {monotone_fit}/{T-1}  (theory: all {T-1}/{T-1})")
spearman_r, spearman_p = stats.spearmanr(np.arange(T), beta_fits)
print(f"  Spearman corr(t, β_fit) = {spearman_r:.4f}  p={spearman_p:.4f}")
if spearman_r > 0.5 and spearman_p < 0.05:
    print("  [PASS] β_t increases with t (insider becomes more aggressive over time).")
else:
    print("  [FAIL] β_t NOT increasing — insider not learning to trade aggressively late.")


# ── 2. Per-period λ: should be constant at λ* = 1.0 ──────────────────────────
print("\n" + "="*72)
print(f"  DIAGNOSTIC 2 – Per-Period MM Strategy λ_t  (theory: constant at {LAMBDA_STAR:.2f})")
print("="*72)
print(f"  {'t':>3}  {'λ*':>6}  {'λ_fit':>8}  {'err%':>8}  {'R²':>7}  {'verdict':>10}")
print("  " + "-"*55)

lambda_fits = []
r2_mm_all   = []

for t in range(T):
    d    = per_step[t]
    y_a  = np.array([s['y']       for s in d])
    dp_a = np.array([s['delta_p'] for s in d])

    l, _, r, _, _ = stats.linregress(y_a, dp_a)
    r2 = r**2
    lambda_fits.append(l)
    r2_mm_all.append(r2)

    err     = 100 * (l - LAMBDA_STAR) / LAMBDA_STAR
    verdict = "PASS" if abs(err) < 25 else "FAIL"
    print(f"  {t+1:>3}  {LAMBDA_STAR:>6.2f}  {l:>8.4f}  {err:>+7.1f}%  {r2:>7.3f}  {verdict:>10}")

lambda_fits = np.array(lambda_fits)
lambda_cv   = lambda_fits.std() / lambda_fits.mean() * 100
lambda_mean = lambda_fits.mean()
lambda_err  = 100 * (lambda_mean - LAMBDA_STAR) / LAMBDA_STAR
print(f"\n  Mean λ_fit = {lambda_mean:.4f}  (theory {LAMBDA_STAR:.2f}, error {lambda_err:+.1f}%)")
print(f"  CV across periods = {lambda_cv:.1f}%  (low = stationary = good)")
if abs(lambda_err) < 25 and lambda_cv < 20:
    print("  [PASS] λ is approximately constant and close to λ*.")
elif lambda_cv < 20:
    print("  [PARTIAL] λ is stationary but level is off — MM has a systematic bias.")
else:
    print("  [FAIL] λ varies across periods and is off-level.")


# ── 3. Sigma_t trajectory: should decrease linearly ──────────────────────────
print("\n" + "="*72)
print("  DIAGNOSTIC 3 – Residual Variance Σ_t = Var(v − p_t)  (theory: linear in t)")
print("  Kyle: Σ_t = Σ_0*(T−t)/T  → full price discovery by round T")
print("="*72)
print(f"  {'t':>3}  {'Σ_t*':>10}  {'Σ_t empirical':>15}  {'ratio':>8}  {'Corr(p,v)':>10}")
print("  " + "-"*55)

sigma_emp = []
for t in range(T):
    d      = per_step[t]
    resid  = np.array([s['v'] - s['p'] for s in d])
    p_arr  = np.array([s['p']          for s in d])
    v_arr  = np.array([s['v']          for s in d])
    se     = resid.var()
    corr   = np.corrcoef(p_arr, v_arr)[0, 1]
    sigma_emp.append(se)
    ratio  = se / sigma_theory[t] if sigma_theory[t] > 0 else np.nan
    print(f"  {t+1:>3}  {sigma_theory[t]:>10.4f}  {se:>15.4f}  {ratio:>8.3f}  {corr:>10.4f}")

sigma_emp = np.array(sigma_emp)

# Check linearity via correlation of empirical Sigma with t
t_arr    = np.arange(1, T + 1)
corr_lin, _ = stats.pearsonr(-t_arr, sigma_emp)  # should be linear DECREASE
print(f"\n  Corr(−t, Σ_t empirical) = {corr_lin:.4f}  (1.0 = perfectly linear decrease)")

monotone_sigma = np.all(np.diff(sigma_emp) <= 0)
pct_revealed   = 100 * (1 - sigma_emp[-1] / Sigma_0)
print(f"  Monotone decrease: {'YES' if monotone_sigma else 'NO (non-monotone steps)'}")
print(f"  % variance revealed at T: {pct_revealed:.1f}%  (theory: 100%)")

if pct_revealed > 80:
    print("  [PASS] Prices reveal most of the private information by T.")
elif pct_revealed > 50:
    print("  [PARTIAL] Moderate price discovery — not fully converged.")
else:
    print("  [FAIL] Poor price discovery — prices barely track true value.")


# ── 4. MM break-even ──────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  DIAGNOSTIC 4 – Market Maker Break-Even  (E[r_mm per episode] ≈ 0)")
print("="*72)

ep_mm_totals = np.array([
    sum(per_step[t][ep]['r_mm'] for t in range(T) if ep < len(per_step[t]))
    for ep in range(N_EVAL)
])
ep_ins_totals = -ep_mm_totals   # zero-sum game: r_insider = -r_mm step-wise

t_stat, p_val = stats.ttest_1samp(ep_mm_totals, 0.0)
print(f"  MM avg episode reward : {ep_mm_totals.mean():+.4f}  std={ep_mm_totals.std():.4f}")
print(f"  t-test E[r_mm]=0  →  t={t_stat:.3f}, p={p_val:.4f}")
if p_val > 0.05 or abs(ep_mm_totals.mean()) < 0.02:
    print("  [PASS] MM breaks even.")
else:
    print(f"  [FAIL] MM has nonzero expected profit — market not in equilibrium.")


# ── 5. Insider profit per period: should peak mid-game ────────────────────────
print("\n" + "="*72)
print("  DIAGNOSTIC 5 – Insider Per-Period Profit Profile")
print("  Theory: insider profit is E[x_t*(v-p_t)] = β_t*(1-λ*β_t)*Σ_{t-1}")
print("="*72)
print(f"  {'t':>3}  {'Theory E[r_t]':>14}  {'Empirical':>10}")
print("  " + "-"*35)

theory_profit_t = []
for t in range(T):
    S   = sigma_theory[t]
    b   = beta_theory[t]
    # E[x_t * (v - p_t)] = β_t * (1 - λ* β_t) * Σ_{t-1}
    # Note: Σ entering period t is sigma_theory[t] = Σ_0*(T-t)/T
    ep_r_theory = b * (1 - LAMBDA_STAR * b) * S / (env.price_std_dev * env.noise_std_dev)
    theory_profit_t.append(ep_r_theory)

for t in range(T):
    emp_r = np.mean([s['r_insider'] for s in per_step[t]])
    print(f"  {t+1:>3}  {theory_profit_t[t]:>14.4f}  {emp_r:>10.4f}")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  OVERALL CONVERGENCE SUMMARY")
print("  (Using correct multi-period Kyle benchmarks)")
print("="*72)

# Recompute verdict with correct benchmarks
beta_errs   = [100*(beta_fits[t] - beta_theory[t])/beta_theory[t] for t in range(T)]
beta_passes  = sum(1 for e in beta_errs if abs(e) < 25)

lambda_errs  = [100*(lambda_fits[t] - LAMBDA_STAR)/LAMBDA_STAR for t in range(T)]
lambda_passes= sum(1 for e in lambda_errs if abs(e) < 25)

checks = {
    f"β_t within 25% of time-varying β_t* ({beta_passes}/{T} periods)":
        beta_passes >= T // 2,
    "β_t increasing with t (Spearman > 0.5, p<0.05)":
        spearman_r > 0.5 and spearman_p < 0.05,
    f"λ stationary (CV={lambda_cv:.1f}% < 20%)":
        lambda_cv < 20,
    f"λ level within 25% of λ*={LAMBDA_STAR} ({lambda_passes}/{T} periods)":
        lambda_passes >= T // 2,
    f"Variance revealed >80% (got {pct_revealed:.1f}%)":
        pct_revealed > 80,
    "MM breaks even":
        p_val > 0.05 or abs(ep_mm_totals.mean()) < 0.02,
    "Insider earns positive expected profit":
        ep_ins_totals.mean() > 0,
}

for desc, ok in checks.items():
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {desc}")

n_pass = sum(checks.values())
print(f"\n  Score: {n_pass}/{len(checks)} checks passed.")
print("="*72)
