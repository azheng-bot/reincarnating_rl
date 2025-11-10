import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Tabula rasa Nature DQN
df = pd.read_csv('nature_dqn_iqm.csv')
axs[0].plot(df['frames'] / 1e6, df['iqm'], label='Nature DQN (Tabula rasa)')
axs[0].fill_between(df['frames'] / 1e6, df['ci_low'], df['ci_high'], alpha=0.3)
axs[0].set_xlabel('Env. Frames (x 10M)')
axs[0].set_ylabel('IQM Normalized Score')
axs[0].legend()
axs[0].set_title('Panel 1')

# Panel 2: Fine-tuning
# for variant in ['finetune_reduced_lr', 'finetune_adam', 'dqn_adam']:
#     df = pd.read_csv(f'{variant}_iqm.csv')
#     axs[1].plot(df['frames'] / 1e6, df['iqm'], label=variant.replace('_', ' ').title())
#     axs[1].fill_between(df['frames'] / 1e6, df['ci_low'], df['ci_high'], alpha=0.3)
# axs[1].set_xlabel('Env. Frames (x 10M)')
# axs[1].legend()
# axs[1].set_title('Panel 2')

# Panel 3: Rainbow
# for variant in ['tabula_rasa_rainbow', 'reincarnate_rainbow']:
#     df = pd.read_csv(f'{variant}_iqm.csv')
#     axs[2].plot(df['frames'] / 1e6, df['iqm'], label=variant.replace('_', ' ').title())
#     axs[2].fill_between(df['frames'] / 1e6, df['ci_low'], df['ci_high'], alpha=0.3)
# axs[2].set_xlabel('Env. Frames (x 10M)')
# axs[2].legend()
# axs[2].set_title('Panel 3')

plt.tight_layout()
plt.savefig('figure1.png')
plt.show()