# prompt_lstm-decision_transformer
Code for Offline Prompt Reinforcement Learning method Based on Feature Extraction

# run
1. Download D4RL datasets：
   http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/
   or run data/download_d4rl_datasets.py
2. Set running tasks and related parameters
   for example:(
    --env=halfcheetah
    --dataset_mode=expert
    --prompt-length=5
    --K=20
    --no-propmt=False
   )
3. run pldt.py
