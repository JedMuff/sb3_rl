from stable_baselines3.common.callbacks import BaseCallback
import csv
import pandas as pd
import os

class CustomCallback(BaseCallback):

    def __init__(self, verbose=0, path='.', file_name='data'):
        super(CustomCallback, self).__init__(verbose)

        self.dir_file_csv = path + file_name + '.csv'
        
        if not(os.path.isdir(path)):
            os.mkdir(path)

        self.policy_step_data = pd.DataFrame()
        self.policy_data = pd.DataFrame()

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

        # Concat with policy_data
        self.policy_data = pd.concat([self.policy_data, df_mean])

        # Take the last value of goal_reached column
        self.policy_data.iloc[-1, self.policy_data.columns.get_loc('goal_reached')] = self.policy_step_data['goal_reached'].iat[-1]

        # Reset data collection
        self.policy_step_data = pd.DataFrame()

    def _on_training_end(self) -> None:
        self.policy_data.to_csv(self.dir_file_csv)