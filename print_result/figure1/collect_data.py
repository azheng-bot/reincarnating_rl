import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_eval_returns(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    times = ea.Scalars('Eval/Steps')  # 或 'Train/Frames' 如果是 frames
    returns = ea.Scalars('Eval/AverageReturns')
    df = pd.DataFrame({'frames': [s.value for s in times], 'returns': [s.value for s in returns]})
    return df

# 示例：收集所有 variants
variants = ['panel1/nature_dqn']
games = ['Breakout', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders', 'BeamRider', 'MsPacman', 'Asterix', 'Riverraid', 'Enduro']  # 假设 10 games
base_path = '../../reincarnating_rl/logs/figure1/'

all_data = {}
for variant in variants:
    variant_data = []
    for game in games:
        for seed in range(1, 51):
            log_dir = os.path.join(base_path, variant, game, f'run_{seed}', 'logs')
            if os.path.exists(log_dir):
                df = extract_eval_returns(log_dir)
                df['game'] = game
                df['seed'] = seed
                variant_data.append(df)
    all_data[variant] = pd.concat(variant_data)
    all_data[variant].to_csv(f'{variant}_data.csv', index=False)