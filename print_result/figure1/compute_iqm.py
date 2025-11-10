import pandas as pd
import numpy as np
import rliable as rl  # 如果没安装，见手动实现
from scipy.stats import bootstrap

# 加载随机分数（从项目 atari_scores.py 复制）

RANDOM_SCORES = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'Asteroids': 719.1,
    'Atlantis': 12850.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'BeamRider': 363.9,
    'Berzerk': 123.7,
    'Bowling': 23.1,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'Centipede': 2090.9,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'Defender': 2874.5,
    'DemonAttack': 152.1,
    'DoubleDunk': -18.6,
    'Enduro': 0.0,
    'FishingDerby': -91.7,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Gravitar': 173.0,
    'Hero': 1027.0,
    'IceHockey': -11.2,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MontezumaRevenge': 0.0,
    'MsPacman': 307.3,
    'NameThisGame': 2292.3,
    'Phoenix': 761.4,
    'Pitfall': -229.4,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'Riverraid': 1338.5,
    'RoadRunner': 11.5,
    'Robotank': 2.2,
    'Seaquest': 68.4,
    'Skiing': -17098.1,
    'Solaris': 1236.3,
    'SpaceInvaders': 148.0,
    'StarGunner': 664.0,
    'Surround': -10.0,
    'Tennis': -23.8,
    'TimePilot': 3568.0,
    'Tutankham': 11.4,
    'UpNDown': 533.4,
    'Venture': 0.0,
    'VideoPinball': 0.0,
    'WizardOfWor': 563.5,
    'YarsRevenge': 3092.9,
    'Zaxxon': 32.5
}
def normalize_scores(df, random_scores):
    for game in df['game'].unique():
        random = random_scores[game]
        dqn_adam_400m = 1.0  # 从 dqn_adam 数据最后均值获取，或硬编码
        mask = df['game'] == game
        df.loc[mask, 'normalized'] = (df.loc[mask, 'returns'] - random) / (dqn_adam_400m - random + 1e-8)
    return df

variants = ['panel1/nature_dqn']
iqm_data = {}
for variant in variants:
    df = pd.read_csv(f'{variant}_data.csv')
    df = normalize_scores(df, RANDOM_SCORES)
    
    # 按 frames 分组，计算 IQM
    frames = sorted(df['frames'].unique())
    iqm_scores = []
    ci_low = []
    ci_high = []
    for f in frames:
        scores = df[df['frames'] == f]['normalized'].values.reshape(10, 50)  # 10 games x 50 seeds
        iqm, ci = rl.get_interval_estimates(rl.metrics.aggregate_iqm, scores, reps=5000)  # bootstrap CI
        iqm_scores.append(iqm)
        ci_low.append(ci[0])
        ci_high.append(ci[1])
    iqm_data[variant] = pd.DataFrame({'frames': frames, 'iqm': iqm_scores, 'ci_low': ci_low, 'ci_high': ci_high})

# 如果无 rliable，手动 IQM
def manual_iqm(scores):
    scores = np.sort(scores)
    q25, q75 = np.percentile(scores, [25, 75])
    return np.mean(scores[(scores >= q25) & (scores <= q75)])

# 手动 bootstrap CI 示例（替换 rl.get_interval_estimates）
def manual_bootstrap(scores, func=manual_iqm, reps=5000, alpha=0.05):
    res = bootstrap((scores.flatten(),), func, n_resamples=reps, confidence_level=1-alpha)
    return res.confidence_interval.low, res.confidence_interval.high