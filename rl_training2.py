#!/usr/bin/env python3
"""
rl_training2.py
---------------
PPO-based self-healing RL agent for Mininet/Faucet SDN environments.

Revisions (reviewer response):
  - Recovery time tracking added to step() and get_recovery_stats()
  - reward_scale parameter added for sensitivity analysis (Comment 4)
  - CLI --csv and --label arguments for multi-topology runs (Comment 2)
  - Results saved as JSON for paper tables

Usage:
  python rl_training2.py --csv rl_features2.csv --label base
  python rl_training2.py --csv rl_features_linear8.csv --label linear8
  python rl_training2.py --csv rl_features_fattree16.csv --label fattree16

Reward sensitivity analysis:
  python rl_training2.py --csv rl_features2.csv --label base --reward_scale 0.5
  python rl_training2.py --csv rl_features2.csv --label base --reward_scale 2.0
"""

import json
import re
import statistics
import argparse

import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class SwitchSelfHealingEnv(gym.Env):
    """
    Gym-compatible RL environment for SDN self-healing.

    State  : 8 Prometheus/Faucet features (CPU, memory, OF stats, ports_down)
    Actions: 0=no-op | 1=reset all ports | 2=restart switch | 3=custom self-heal
    Reward : severity-weighted, with heal bonuses and unnecessary-action penalties
    """

    def __init__(self, csv_path, reward_scale=1.0):
        super().__init__()

        self.data          = pd.read_csv(csv_path).fillna(0)
        self.current_step  = 0
        self.reward_scale  = reward_scale

        # ── Recovery time tracking ────────────────────────────────────────────
        self.fault_start_step = None
        self.recovery_times   = []

        # ── Feature columns ───────────────────────────────────────────────────
        self.feature_cols = [
            'process_cpu_seconds_total',
            'process_resident_memory_bytes',
            'process_virtual_memory_bytes',
            'of_flowmsgs_sent_total',
            'of_errors_total',
            'of_dp_connections_total',
            'of_dp_disconnections_total',
            'ports_down_count',
        ]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32
        )

        # 0=no-op | 1=reset ports | 2=restart switch | 3=self-heal
        self.action_space = spaces.Discrete(4)

        # ── Reward table (severity-proportional, see paper Section 4.3.2) ─────
        self.fault_weights = {
            'LINK_FAULT':   {'penalty': 10, 'heal_bonus': 12},
            'PORT_FAULT':   {'penalty':  7, 'heal_bonus': 10},
            'FLOW_ERROR':   {'penalty':  5, 'heal_bonus':  8},
            'SOFT_WARNING': {'penalty':  2, 'heal_bonus':  2},
            'TEMP_WARNING': {'penalty':  1, 'heal_bonus':  1},
            'NONE':         {'penalty':  0, 'heal_bonus':  0},
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def reset(self):
        self.current_step     = 0
        self.fault_start_step = None
        self.recovery_times   = []
        return self._get_obs()

    def _get_obs(self):
        return (self.data
                .loc[self.current_step, self.feature_cols]
                .values
                .astype(np.float32))

    def _parse_fault_and_duration(self, last_fault_str):
        """
        Returns (fault_type, duration_seconds).
        Matches fault keywords in priority order; parses 'duration=XX.Xs'
        if present. Falls back to SOFT_WARNING for unrecognised strings.
        """
        if pd.isnull(last_fault_str) or str(last_fault_str).strip() == '':
            return 'NONE', 0.0

        lf = last_fault_str.upper()

        if 'FAUCET_RELOAD' in lf:
            return 'FAUCET_RELOAD', 0.0

        for key in self.fault_weights:
            if key != 'NONE' and key in lf:
                match    = re.search(r'duration=([\d\.]+)S', lf)
                duration = float(match.group(1)) if match else 0.0
                return key, duration

        match    = re.search(r'duration=([\d\.]+)S', lf)
        duration = float(match.group(1)) if match else 0.0
        return 'SOFT_WARNING', duration

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, action):
        row        = self.data.loc[self.current_step]
        reward     = 0.0
        last_fault = str(row.get('last_fault', '')).upper()

        # ── Fault onset tracking ──────────────────────────────────────────────
        fault_active = ('FAULT' in last_fault or 'RELOAD' in last_fault)
        if fault_active and self.fault_start_step is None:
            self.fault_start_step = self.current_step

        # ── Successful healing detection ──────────────────────────────────────
        healing_taken = action in [1, 2, 3]
        healed = (
            self.fault_start_step is not None
            and healing_taken
            and row.get('ports_down_count', 1) == 0
            and row.get('of_errors_total',  1) == 0
        )
        if healed:
            recovery_steps = self.current_step - self.fault_start_step
            self.recovery_times.append(recovery_steps)
            self.fault_start_step = None

        # Reset tracker when fault clears without agent action
        if not fault_active and self.fault_start_step is not None:
            self.fault_start_step = None

        # ── Reward logic ──────────────────────────────────────────────────────
        if 'FAUCET_RELOAD' in last_fault:
            reward += 20
            print(f"[REWARD] FAUCET_RELOAD at step {self.current_step}: +20")
        else:
            reward -= row['ports_down_count'] * 2
            reward -= row['of_errors_total']

            fault_type, fault_duration = self._parse_fault_and_duration(
                row.get('last_fault', ''))
            weights = self.fault_weights.get(
                fault_type, self.fault_weights['SOFT_WARNING'])

            if fault_type != 'NONE':
                reward -= weights['penalty']
                reward -= fault_duration

            if fault_type != 'NONE' and action in [1, 2, 3]:
                reward += weights['heal_bonus']

            if (fault_type == 'NONE'
                    and row.get('ports_down_count', 0) == 0
                    and row.get('of_errors_total',  0) == 0
                    and action == 0):
                reward += 2

            if fault_type == 'NONE' and action in [1, 2, 3]:
                reward -= 1

        # Apply reward scale (for sensitivity analysis)
        reward *= self.reward_scale

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs  = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, reward, done, {}

    # ── Recovery stats ────────────────────────────────────────────────────────

    def get_recovery_stats(self):
        """
        Returns dict with mean/std recovery time in steps and seconds.
        Each timestep = 10 seconds (scrape interval).
        Returns None if no healing events were recorded.
        """
        if not self.recovery_times:
            return None
        secs = [s * 10 for s in self.recovery_times]
        return {
            'n_events'   : len(self.recovery_times),
            'mean_steps' : round(statistics.mean(self.recovery_times), 2),
            'std_steps'  : round(statistics.stdev(self.recovery_times), 2)
                           if len(self.recovery_times) > 1 else 0.0,
            'mean_secs'  : round(statistics.mean(secs), 1),
            'std_secs'   : round(statistics.stdev(secs), 1)
                           if len(self.recovery_times) > 1 else 0.0,
            'all_steps'  : self.recovery_times,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reward Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity_analysis(csv_path, timesteps=10_000):
    """
    Trains PPO under three reward scales (x0.5, x1.0, x2.0) and
    reports mean episode reward and mean recovery time for each.
    Saves results to reward_sensitivity.csv.
    """
    import pandas as pd

    scales  = [0.5, 1.0, 2.0]
    results = []

    for scale in scales:
        print(f"\n[SENSITIVITY] Training with reward_scale={scale} ...")
        env   = SwitchSelfHealingEnv(csv_path, reward_scale=scale)
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=timesteps)

        obs = env.reset()
        total_reward = 0.0
        steps        = 0
        for _ in range(len(env.data)):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total_reward += r
            steps        += 1
            if done:
                break

        stats = env.get_recovery_stats()
        results.append({
            'reward_scale'       : f'x{scale}',
            'mean_episode_reward': round(total_reward / max(steps, 1), 2),
            'mean_recovery_steps': stats['mean_steps'] if stats else 'N/A',
            'mean_recovery_secs' : stats['mean_secs']  if stats else 'N/A',
            'n_healing_events'   : stats['n_events']   if stats else 0,
        })
        print(f"[SENSITIVITY] scale={scale} | reward={results[-1]['mean_episode_reward']} "
              f"| recovery={results[-1]['mean_recovery_secs']}s")

    df = pd.DataFrame(results)
    df.to_csv('reward_sensitivity.csv', index=False)
    print("\n[SENSITIVITY] Results saved to reward_sensitivity.csv")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate PPO self-healing agent for SDN.')
    parser.add_argument('--csv',          default='rl_features2.csv',
                        help='Path to feature CSV (default: rl_features2.csv)')
    parser.add_argument('--label',        default='base',
                        help='Topology label: base | linear8 | fattree16')
    parser.add_argument('--timesteps',    default=10_000, type=int,
                        help='PPO training timesteps (default: 10000)')
    parser.add_argument('--reward_scale', default=1.0,    type=float,
                        help='Reward scale factor for sensitivity analysis')
    parser.add_argument('--sensitivity',  action='store_true',
                        help='Run reward sensitivity analysis across x0.5/x1.0/x2.0')
    args = parser.parse_args()

    # ── Optional: reward sensitivity analysis ─────────────────────────────────
    if args.sensitivity:
        run_sensitivity_analysis(args.csv, timesteps=args.timesteps)
        exit(0)

    # ── Standard training run ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training PPO Agent | topology={args.label} | scale={args.reward_scale}")
    print(f"{'='*60}")

    env = SwitchSelfHealingEnv(args.csv, reward_scale=args.reward_scale)

    # Quick sanity check on environment
    print("[INFO] Environment created.")
    obs = env.reset()
    print(f"[INFO] Observation space : {env.observation_space}")
    print(f"[INFO] Action space      : {env.action_space}")
    print(f"[INFO] Dataset rows      : {len(env.data)}")
    print(f"[INFO] Initial obs       : {obs}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = PPO(
        'MlpPolicy', env,
        verbose=1,
        tensorboard_log='./ppo_switch_tensorboard'
    )
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=f'PPO_FAUCET_HEAL_{args.label}_scale{args.reward_scale}'
    )
    model_path = f'ppo_switch_agent_{args.label}'
    model.save(model_path)
    print(f"\n[INFO] Model saved to {model_path}.zip")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating trained agent on full dataset ...")
    obs          = env.reset()
    total_reward = 0.0
    step_count   = 0

    for i in range(len(env.data)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count   += 1
        if i < 5:
            print(f"  [EVAL {i}] action={action}  reward={reward:.2f}  done={done}")
        if done:
            break

    # ── Recovery stats ────────────────────────────────────────────────────────
    stats = env.get_recovery_stats()
    print(f"\n{'='*60}")
    print(f"  Recovery Time Stats  [{args.label}]")
    print(f"{'='*60}")
    if stats:
        print(json.dumps(stats, indent=2))
        out_path = f'recovery_stats_{args.label}.json'
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] Stats saved to {out_path}")
    else:
        print("[WARN] No healing events recorded during evaluation.")

    print(f"\n[INFO] Total eval reward : {total_reward:.2f}")
    print(f"[INFO] Total eval steps  : {step_count}")
    print(f"\n{'='*60}")
    print("  Script Finished")
    print(f"{'='*60}\n")
