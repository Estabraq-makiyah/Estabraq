# ── Add these imports at the top ──────────────────────────────────────────────
import json, statistics

# ── Inside SwitchSelfHealingEnv.__init__() add: ───────────────────────────────
self.fault_start_step   = None   # timestep when current fault began
self.recovery_times     = []     # list of (steps_to_heal) per fault event

# ── Replace the existing step() method body with this: ────────────────────────
def step(self, action):
    row        = self.data.loc[self.current_step]
    reward     = 0.0
    last_fault = str(row.get('last_fault', '')).upper()

    # ── Track fault onset ────────────────────────────────────────────────────
    fault_active = ('FAULT' in last_fault or 'RELOAD' in last_fault)
    if fault_active and self.fault_start_step is None:
        self.fault_start_step = self.current_step          # fault begins

    # ── Detect successful healing ─────────────────────────────────────────────
    healing_action_taken = (action in [1, 2, 3])
    healed = (
        self.fault_start_step is not None
        and healing_action_taken
        and row.get('ports_down_count', 1) == 0
        and row.get('of_errors_total', 1)  == 0
    )
    if healed:
        recovery_steps = self.current_step - self.fault_start_step
        self.recovery_times.append(recovery_steps)
        self.fault_start_step = None                       # reset tracker

    # ── Reset fault tracker when network returns healthy without agent ────────
    if not fault_active and self.fault_start_step is not None:
        self.fault_start_step = None

    # ── Original reward logic (unchanged) ────────────────────────────────────
    if 'FAUCET_RELOAD' in last_fault:
        reward += 20
    else:
        reward -= row['ports_down_count'] * 2
        reward -= row['of_errors_total']
        fault_type, fault_duration = self.parse_fault_and_duration(
            row.get('last_fault', ''))
        weights = self.fault_weights.get(fault_type, self.fault_weights['SOFT_WARNING'])
        if fault_type != 'NONE':
            reward -= weights['penalty']
            reward -= fault_duration
        if fault_type != 'NONE' and action in [1, 2, 3]:
            reward += weights['heal_bonus']
        if fault_type == 'NONE' and row.get('ports_down_count',0) == 0 \
                and row.get('of_errors_total',0) == 0 and action == 0:
            reward += 2
        if fault_type == 'NONE' and action in [1, 2, 3]:
            reward -= 1

    self.current_step += 1
    done = self.current_step >= len(self.data) - 1
    obs  = self.get_obs() if not done else np.zeros_like(self.get_obs())
    return obs, reward, done, {}

def get_recovery_stats(self):
    """Return mean, std, and per-event recovery times (in steps and seconds)."""
    if not self.recovery_times:
        return None
    secs = [s * 10 for s in self.recovery_times]   # each step = 10 s
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

# ── Replace __main__ block: ───────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',   default='rl_features2.csv')
    parser.add_argument('--label', default='base',
                        help='topology label: base | linear8 | fattree16')
    args = parser.parse_args()

    env   = SwitchSelfHealingEnv(args.csv)
    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log='./ppo_switch_tensorboard')
    model.learn(total_timesteps=10_000,
                tb_log_name=f'PPO_FAUCET_HEAL_{args.label}')
    model.save(f'ppo_switch_agent_{args.label}')

    # ── Evaluate and record recovery time ────────────────────────────────────
    obs = env.reset()
    for _ in range(len(env.data)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break

    stats = env.get_recovery_stats()
    print(f"\n=== Recovery Time Stats [{args.label}] ===")
    print(json.dumps(stats, indent=2))

    # Save to file for paper table
    with open(f'recovery_stats_{args.label}.json', 'w') as f:
        json.dump(stats, f, indent=2)