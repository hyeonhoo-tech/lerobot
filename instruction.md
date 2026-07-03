#For robot dataset
python lerobot/scripts/control_robot.py --robot.type=trossen_ai_stationary --robot.max_relative_target=null --control.type=record --control.fps=30 --control.single_task="Open the drawer" --control.repo_id=${HF_USER}/trial_test_0 --control.tags='["tutorial"]' --control.warmup_time_s=5 --control.episode_time_s=600 --control.reset_time_s=10 --control.num_episodes=1  --control.push_to_hub=true --control.urdf_path stationary_ai.urdf


#For human dataset
python lerobot/scripts/control_robot.py --robot.type=trossen_ai_stationary --robot.max_relative_target=null --control.type=record --control.fps=30 --control.single_task="Open the drawer" --control.repo_id=${HF_USER}/play_human_v3_1 --control.tags='["tutorial"]' --control.warmup_time_s=5 --control.episode_time_s=600 --control.reset_time_s=10 --control.num_episodes=1  --control.push_to_hub=true --robot.follower_arms='{}' --robot.leader_arms='{}'


#For robot dataset - constraint
python lerobot/scripts/control_robot.py --robot.type=trossen_ai_stationary --robot.max_relative_target=null --control.type=record --control.fps=30 --control.single_task="Open the drawer" --control.repo_id=${HF_USER}/trial_test_0 --control.tags='["tutorial"]' --control.warmup_time_s=5 --control.episode_time_s=600 --control.reset_time_s=10 --control.num_episodes=1  --control.push_to_hub=true --control.urdf_path stationary_ai.urdf --robot.lock_z --robot.lock_wrist_joints




#New
python lerobot/scripts/control_robot.py   --robot.type=trossen_ai_stationary   --robot.max_relative_target=0.1   --robot.lock_z=true   --robot.locked_z=0.1   --robot.lock_orientation=true   --robot.locked_pitch_offset_deg=15   --robot.constraint_soft_start_s=8.0   --control.type=record   --control.urdf_path=stationary_ai.urdf   --control.fps=30   --control.single_task="Open the drawer"   --control.repo_id=${HF_USER}/test  --control.tags='["tutorial"]'   --control.warmup_time_s=15   --control.episode_time_s=300   --control.reset_time_s=15   --control.num_episodes=1   --control.push_to_hub=true --control.tolerance_s 0.01


#robot_can
python lerobot/scripts/control_robot.py   --robot.type=trossen_ai_stationary   --robot.max_relative_target=0.1   --robot.lock_z=false   --robot.lock_orientation=true   --robot.locked_pitch_offset_deg=15   --robot.constraint_soft_start_s=8.0   --control.type=record   --control.urdf_path=stationary_ai.urdf   --control.fps=30   --control.single_task="Place can"   --control.repo_id=${HF_USER}/play_robot_can_test  --control.tags='["tutorial"]'   --control.warmup_time_s=15   --control.episode_time_s=600   --control.reset_time_s=15   --control.num_episodes=1   --control.push_to_hub=true --control.tolerance_s 0.01

#human_Can
python lerobot/scripts/control_robot.py --robot.type=trossen_ai_stationary --robot.max_relative_target=null --control.type=record --control.fps=30 --control.single_task="Place can" --control.repo_id=${HF_USER}/play_human_can_test --control.tags='["tutorial"]' --control.warmup_time_s=5 --control.episode_time_s=600 --control.reset_time_s=10 --control.num_episodes=1  --control.push_to_hub=true --robot.follower_arms='{}' --robot.leader_arms='{}'