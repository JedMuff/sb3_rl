from stable_baselines3.common.callbacks import BaseCallback
import csv
import pandas as pd
import os
import numpy as np

class CustomCallback(BaseCallback):

    def __init__(self, verbose=0, path='.', file_name='data'):
        super(CustomCallback, self).__init__(verbose)

        self.path = path
        self.file_name = file_name

        if not(os.path.isdir(path)):
            os.mkdir(path)

        # Data collected on step and reset on rollout
        self.step_data = pd.DataFrame()

        # Data collected on rollout and saved on training end
        self.mean_rollout_data = pd.DataFrame()
        self.std_rollout_data = pd.DataFrame()
        self.rollout_goal_reached_accuracy = pd.DataFrame()

    def _on_step(self) -> bool:
        all_data = self.training_env.env_method('get_step_info')

        for idx, cpu_data in enumerate(all_data):
            state, reward, info = cpu_data

            data = info.copy()
            data['net_reward'] = reward
            
            self.step_data = pd.concat([self.step_data, pd.DataFrame([data])], ignore_index=True)

        return True

    def _on_rollout_end(self) -> None:

        # Remove goal_reached_col
        goal_reached_col = self.step_data.pop('goal_reached')

        # Average columns
        df_mean = pd.DataFrame(self.step_data.mean()).T
        df_std = pd.DataFrame(self.step_data.std()).T

        # preprocess goal_reached_col
        goal_reached_col.dropna(inplace=True)
        total_attempts = goal_reached_col.shape[0]
        hits = goal_reached_col.sum()
        percentage_goals_reached = pd.DataFrame([{'percentage_goals_reached' : hits / total_attempts}])

        # Concat with rollout_data
        self.mean_rollout_data = pd.concat([self.mean_rollout_data, df_mean])
        self.std_rollout_data = pd.concat([self.std_rollout_data, df_std])
        self.rollout_goal_reached_accuracy = pd.concat([self.rollout_goal_reached_accuracy, percentage_goals_reached])

        # Reset data collection
        self.step_data = pd.DataFrame()

    def _on_training_end(self) -> None:
        self.mean_rollout_data.to_csv(self.path + 'mean_'+self.file_name + '.csv')
        self.std_rollout_data.to_csv(self.path + 'std_'+self.file_name + '.csv')
        self.rollout_goal_reached_accuracy.to_csv(self.path + 'goals_'+self.file_name + '.csv')
