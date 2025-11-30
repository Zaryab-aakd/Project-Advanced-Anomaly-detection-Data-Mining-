import os
import pandas as pd
import numpy as np

class SMDLoader:
    def __init__(self, base_path='OmniAnomaly/ServerMachineDataset'):
        self.base_path = base_path

    def load_machine(self, machine_name='machine-1-1'):
        print(f"--> [Data Loader] Loading data for: {machine_name}")
        train_path = os.path.join(self.base_path, 'train', f'{machine_name}.txt')
        test_path = os.path.join(self.base_path, 'test', f'{machine_name}.txt')
        label_path = os.path.join(self.base_path, 'test_label', f'{machine_name}.txt')

        try:
            train_df = pd.read_csv(train_path, header=None)
            test_df = pd.read_csv(test_path, header=None)
            label_df = pd.read_csv(label_path, header=None)
            return train_df.values, test_df.values, label_df.values.flatten()
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None
