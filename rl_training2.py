import gym
from gym import spaces
import numpy as np
import pandas as pd
import re
from stable_baselines3 import PPO

class SwitchSelfHealingEnv(gym.Env):
    """
    RL environment for self-healing switching network using rl_features2.csv.
    Rewards the agent whenever FAUCET_RELOAD appears in last_fault (successful healing).
    All other advanced reward logic as before.
    """
    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.current_step = 0

        # Features used as RL state (edit if your columns differ!)
        self.feature_cols = [
            'process_cpu_seconds_total',
            'process_resident_memory_bytes',
            'process_virtual_memory_bytes',
            'of_flowmsgs_sent_total',
            'of_errors_total',
            'of_dp_connections_total',
            'of_dp_disconnections_total',
            'ports_down_count'
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )
        # Actions: 0=do nothing, 1=reset all ports, 2=restart switch, 3=custom self-heal
        self.action_space = spaces.Discrete(4)

        # Fault reward/penalty setup
        self.fault_weights = {
            'LINK_FAULT':      {'penalty': 10, 'heal_bonus': 12},
            'PORT_FAULT':      {'penalty': 7,  'heal_bonus': 10},
            'FLOW_ERROR':      {'penalty': 5,  'heal_bonus': 8},
            'SOFT_WARNING':    {'penalty': 2,  'heal_bonus': 2},
            'TEMP_WARNING':    {'penalty': 1,  'heal_bonus': 1},
            'NONE':            {'penalty': 0,  'heal_bonus': 0},
        }

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs = self.data.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        return obs

    def _parse_fault_and_duration(self, last_fault_str):
        """
        Returns (fault_type, duration)
        - If FAUCET_RELOAD is present, treat as FAUCET_RELOAD type (for reward).
        - If no fault or empty, treat as NONE.
        - Otherwise, match by keywords in fault_weights.
        - Parses duration from string if present (duration=XX.Xs).
        """
        if pd.isnull(last_fault_str) or str(last_fault_str).strip() == '':
            return 'NONE', 0.0
        lf = last_fault_str.upper()
        if 'FAUCET_RELOAD' in lf:
            return 'FAUCET_RELOAD', 0.0
        for key in self.fault_weights:
            if key != 'NONE' and key in lf:
                match = re.search(r'duration=([\d\.]+)S', lf)
                duration = float(match.group(1)) if match else 0.0
                return key, duration
        match = re.search(r'duration=([\d\.]+)S', lf)
        duration = float(match.group(1)) if match else 0.0
        return 'SOFT_WARNING', duration

    def step(self, action):
        row = self.data.loc[self.current_step]
        reward = 0.0

        last_fault = str(row.get('last_fault', '')).upper()
        
        # Direct reward for FAUCET_RELOAD
        if 'FAUCET_RELOAD' in last_fault:
            reward = 20  # Large positive reward for successful healing
            print(f"[REWARD] FAUCET_RELOAD at step {self.current_step}: +20 reward")
        else:
            # Standard penalties for ongoing faults and errors
            reward -= row['ports_down_count'] * 2
            reward -= row['of_errors_total']

            # Use last_fault for advanced reward shaping
            fault_type, fault_duration = self._parse_fault_and_duration(row.get('last_fault', ''))
            weights = self.fault_weights.get(fault_type, self.fault_weights['SOFT_WARNING'])

            # Penalty for faults except NONE (duration-weighted)
            if fault_type != 'NONE':
                reward -= weights['penalty']
                reward -= fault_duration  # More penalty for longer faults

            # Extra bonus if healing actions are taken for non-trivial faults
            if fault_type != 'NONE' and action in [1, 2, 3]:
                reward += weights['heal_bonus']

            # If everything is healthy and agent does nothing, give small bonus
            if fault_type == 'NONE' and row['ports_down_count'] == 0 and row['of_errors_total'] == 0 and action == 0:
                reward += 2

            # Small penalty for "reset" actions if no fault present (to discourage unnecessary resets)
            if fault_type == 'NONE' and action in [1, 2, 3]:
                reward -= 1

        # Step forward
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, reward, done, {}

if __name__ == '__main__':
    print("===== RL Environment Testing =====")
    env = SwitchSelfHealingEnv('rl_features2.csv')
    print("[INFO] Environment created.")

    # Manual test: print a few random actions and their results
    obs = env.reset()
    print("[INFO] Initial observation:", obs)
    for i in range(5):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(f"[STEP {i}] Action: {action}, Reward: {reward}, Done: {done}")
        print(f"[STEP {i}] Next observation: {next_obs}")
        if done:
            print("[INFO] End of data reached during manual stepping.")
            break

    # Train the PPO agent with TensorBoard logging enabled
    print("[INFO] Starting RL training...")
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_switch_tensorboard/")
    model.learn(total_timesteps=10000, tb_log_name="PPO_FAUCET_HEAL")
    print("[INFO] Training finished. Saving model...")
    model.save("ppo_switch_agent2")

    # Evaluate the trained agent
    print("[INFO] Evaluating the trained agent...")
    obs = env.reset()
    for i in range(5):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"[EVAL {i}] Agent Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("[INFO] End of data reached during agent evaluation.")
            break

    print("===== Script Finished =====")
