from stable_baselines3.common.callbacks import BaseCallback
import csv
import pandas as pd
import os
import numpy as np

class CustomCallback(BaseCallback):

    def __init__(self, verbose=0, path='.', file_name='data'):
        super(CustomCallback, self).__init__(verbose)

        self.dir_file_csv = path + file_name + '.csv'
        
        if not(os.path.isdir(path)):
            os.mkdir(path)

        self.policy_step_data = pd.DataFrame()
        self.policy_data = pd.DataFrame()
        self.prev_val = 0

    def _on_step(self) -> bool:
        all_data = self.training_env.env_method('get_step_info')

        for idx, cpu_data in enumerate(all_data):
            state, reward, info = cpu_data

            data = info.copy()
            data['net_reward'] = reward
            
            self.policy_step_data = pd.concat([self.policy_step_data, pd.DataFrame([data])], ignore_index=True)

        return True

    def _on_rollout_end(self) -> None:

        # Average columns
        df_mean = pd.DataFrame(self.policy_step_data.mean()).T
        goals_reached = np.sum(self.training_env.get_attr('goal_reached_count'))
        df_mean['goals_reached'] = goals_reached - self.prev_val

        # Concat with policy_data
        self.policy_data = pd.concat([self.policy_data, df_mean])

        # Reset data collection
        self.policy_step_data = pd.DataFrame()
        self.prev_val = goals_reached

    def _on_training_end(self) -> None:
        self.policy_data.to_csv(self.dir_file_csv)