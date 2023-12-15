import pandas as pd
from pathlib import Path

tsv_filepath = Path('/mnt/beegfs/robotlearn/xbie/VoiceBankDemand/test.tsv')
metadata = pd.read_csv(tsv_filepath)

idx = 0
mix_info = metadata.iloc[idx]
breakpoint()