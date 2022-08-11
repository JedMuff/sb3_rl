from stable_baselines3.common.callbacks import BaseCallback
import csv
import pandas as pd
import os

class CustomCallback1(BaseCallback):

    def __init__(self, verbose=0, path='.', step_file_name='step_data_cpu', policy_file_name='pol_data_cpu'):
        super(CustomCallback1, self).__init__(verbose)
        self.field_names = ["task_reward", "velocity_penalty", "position_penalty",
                            "acceleration_penalty","velocity_over_limit","position_over_limit",
                            "acceleration_over_limit","control_penalty", "net_reward", 
                            "ts", "n_calls"]
                
        self.path = path
        self.step_file_name = step_file_name
        self.step_dir_file = self.path + self.step_file_name

        self.pol_file_name = policy_file_name
        self.pol_dir_file = self.path + self.pol_file_name

    def _on_training_start(self) -> None:
        self.init_csv()
        self.start_policy_idx = 0

    def init_csv(self):
        for cpu in range(self.training_env.num_envs):
            file_name = self.step_dir_file + str(cpu) + '.csv'
            with open(file_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.field_names)

        file_name = self.pol_dir_file + '.csv'
        with open(file_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.field_names)

    def _on_step(self) -> bool:
        data = self.training_env.env_method('get_step_info')

        for idx, cpu_data in enumerate(data):
            state, reward, info = cpu_data

            data = info.copy()
            data['net_reward'] = reward
            data['ts'] = self.num_timesteps
            data['n_calls'] = self.n_calls
            
            file_name = self.step_dir_file + str(idx) + '.csv'
            self.append_dict_as_row(file_name, data)

        return True

    def append_dict_as_row(self, file_name, dict):
        with open(file_name, 'a+', newline='') as write_obj:
            dict_writer = csv.DictWriter(write_obj, fieldnames=self.field_names)
            dict_writer.writerow(dict)

    def _on_rollout_end(self) -> None:
        self.finish_policy_idx = self.n_calls

        # Get all data frames from each cpu in a list
        dfs = [pd.read_csv(self.step_dir_file + str(cpu) + '.csv') for cpu in range(self.training_env.num_envs)]

        # crop range
        cropped_dfs = [df.iloc[self.start_policy_idx:self.finish_policy_idx] for df in dfs]

        # add all data frames together
        cumsum_df = cropped_dfs.pop(0)
        for df in cropped_dfs:
            cumsum_df = cumsum_df.add(df)
            
        # Average columns
        df_mean = pd.DataFrame(df.mean()).T

        # append to policy csv file
        df_mean.to_csv(self.pol_dir_file + '.csv', mode='a', index=False, header=False)

        self.start_policy_idx = self.finish_policy_idx - 1


class CustomCallback2(BaseCallback):

    def __init__(self, verbose=0, path='.', file_name='data'):
        super(CustomCallback2, self).__init__(verbose)
        self.field_names = ["task_reward", "velocity_penalty", "position_penalty",
                            "acceleration_penalty","velocity_over_limit","position_over_limit",
                            "acceleration_over_limit","control_penalty", "net_reward"]
                
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

        self.policy_step_data = pd.DataFrame()
        
    def _on_training_end(self) -> None:
        self.policy_data.to_csv(self.dir_file_csv)