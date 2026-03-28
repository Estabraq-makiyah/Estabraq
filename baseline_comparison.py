#!/usr/bin/env python3
"""
Quantitative baseline comparison for the paper.
Runs three agents on the same rl_features2.csv data:
  1. No-Healing  (no action taken — represents unmanaged SDN)
  2. Rule-Based  (reactive: always reset ports on any fault — approximates FRRL)
  3. PPO Agent   (proposed framework)
Measures: fault recovery time (steps), packet loss proxy, OF error rate.
"""

import numpy as np
import pandas as pd
import re, json, statistics
from stable_baselines3 import PPO

CSV_PATH   = 'rl_features2.csv'
MODEL_PATH = 'ppo_switch_agent2'
STEP_SECS  = 10   # each timestep = 10 seconds

# ── Load data ────────────────────────────────────────────────────────────────
data = pd.read_csv(CSV_PATH).fillna(0)
data['last_fault'] = data['last_fault'].astype(str).str.upper()

FAULT_KEYWORDS = ['LINK_FAULT', 'PORT_FAULT', 'FLOW_ERROR',
                  'TRUNK_FAULT', 'FAUCET_RELOAD']

def is_fault(row):
    return any(kw in str(row['last_fault']) for kw in FAULT_KEYWORDS)

def is_healthy(row):
    return (row['ports_down_count'] == 0 and
            row['of_errors_total']  == 0 and
            not is_fault(row))

# ── Metric collectors ────────────────────────────────────────────────────────
def run_agent(agent_fn, label):
    """
    agent_fn(row, step) -> action (0=noop, 1=reset, 2=restart, 3=heal)
    Returns dict of metrics.
    """
    recovery_times   = []
    fault_start      = None
    unresolved_steps = 0
    error_accumulate = 0
    ports_down_accum = 0
    total_steps      = len(data)

    for step, row in data.iterrows():
        action = agent_fn(row, step)
        fault  = is_fault(row)
        healed = (
            fault_start is not None
            and action in [1, 2, 3]
            and row['ports_down_count'] == 0
            and row['of_errors_total']  == 0
        )

        if fault and fault_start is None:
            fault_start = step
        if healed and fault_start is not None:
            recovery_times.append(step - fault_start)
            fault_start = None
        if fault and fault_start is not None:
            unresolved_steps += 1
        if not fault and fault_start is not None:
            fault_start = None   # self-resolved

        error_accumulate += row['of_errors_total']
        ports_down_accum += row['ports_down_count']

    mean_rt   = round(statistics.mean(recovery_times), 2)  if recovery_times else 'N/A'
    std_rt    = round(statistics.stdev(recovery_times), 2) if len(recovery_times) > 1 else 0.0
    mean_secs = round(mean_rt * STEP_SECS, 1)              if mean_rt != 'N/A' else 'N/A'

    # Packet loss proxy: fraction of steps with ports down
    pkt_loss_pct = round(100.0 * ports_down_accum / total_steps, 2)
    # OF error rate: average errors per step
    of_err_rate  = round(error_accumulate / total_steps, 4)
    # Fault exposure time (steps unresolved / total)
    fault_exp    = round(100.0 * unresolved_steps / total_steps, 2)

    return {
        'agent'              : label,
        'n_recoveries'       : len(recovery_times),
        'mean_recovery_steps': mean_rt,
        'std_recovery_steps' : std_rt,
        'mean_recovery_secs' : mean_secs,
        'packet_loss_pct'    : pkt_loss_pct,
        'of_error_rate'      : of_err_rate,
        'fault_exposure_pct' : fault_exp,
    }

# ── Agent 1: No-Healing (always no-op) ───────────────────────────────────────
def no_heal_agent(row, step):
    return 0   # always do nothing

# ── Agent 2: Rule-Based Reactive (approximates FRRL-style reactive recovery) ─
def rule_based_agent(row, step):
    """
    Mimics a reactive rule-based approach:
    - If link/trunk fault → restart switch (action 2)
    - If port fault       → reset ports    (action 1)
    - If flow error       → custom heal    (action 3)
    - Otherwise           → no-op          (action 0)
    """
    lf = str(row['last_fault'])
    if 'LINK_FAULT' in lf or 'TRUNK_FAULT' in lf:
        return 2
    elif 'PORT_FAULT' in lf:
        return 1
    elif 'FLOW_ERROR' in lf:
        return 3
    elif row['ports_down_count'] > 0:
        return 1
    return 0

# ── Agent 3: Trained PPO ──────────────────────────────────────────────────────
from rl_training2 import SwitchSelfHealingEnv
env   = SwitchSelfHealingEnv(CSV_PATH)
model = PPO.load(MODEL_PATH, env=env)
obs   = env.reset()

ppo_actions = {}
for step in range(len(data)):
    action, _ = model.predict(obs, deterministic=True)
    ppo_actions[step] = int(action)
    obs, _, done, _ = env.step(action)
    if done:
        break

def ppo_agent(row, step):
    return ppo_actions.get(step, 0)

# ── Run all three ─────────────────────────────────────────────────────────────
results = [
    run_agent(no_heal_agent,   'No-Healing Baseline'),
    run_agent(rule_based_agent,'Rule-Based Reactive (FRRL-style)'),
    run_agent(ppo_agent,       'Proposed PPO Framework'),
]

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv('baseline_comparison_results.csv', index=False)
print("\nSaved: baseline_comparison_results.csv")