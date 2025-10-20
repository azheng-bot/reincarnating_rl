
# windows可运行
python -um reincarnating_rl.train    --agent="qdagger_rainbow"    --gin_files=".\configs\qdagger_rainbow.gin"    --base_dir=".\tmp\qdagger_rainbow"    --teacher_checkpoint_dir=".\teacher\Breakout\1"    --teacher_checkpoint_number=399    --run_number=1    --gin_bindings="Runner.evaluation_steps=10"    --gin_bindings="JaxDQNAgent.min_replay_history = 64"    --gin_bindings="RainbowAgent.num_actions=4"  --alsologtostderr  

# linux
python -um reincarnating_rl.train    --agent qdagger_rainbow    --gin_files reincarnating_rl/configs/qdagger_rainbow.gin    --base_dir /tmp/qdagger_rainbow       --run_number=1    --gin_bindings="Runner.evaluation_steps=10"    --gin_bindings="JaxDQNAgent.min_replay_history = 64"    --alsologtostderr
python -um reincarnating_rl.train    --agent qdagger_rainbow    --gin_files reincarnating_rl/configs/qdagger_rainbow.gin    --base_dir /tmp/qdagger_rainbow     --teacher_checkpoint_dir ./reincarnating_rl/teacher_ckpt/Breakout/1    --teacher_checkpoint_number=399      --run_number=1    --gin_bindings="Runner.evaluation_steps=10"    --gin_bindings="JaxDQNAgent.min_replay_history = 64"    --alsologtostderr