# Toy-10
python ./NN_train.py --env Toy-10 --beta 0.01 --tau 1.0 --dim_emb 8 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# Toy-100
python ./NN_train.py --env Toy-100_design --beta 0.01 --tau 1.0 --dim_emb 100 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy
python ./NN_train.py --env Toy-100_design --beta 0.1 --tau 1.0 --lr 0.5 --dim_emb 100 --num_train 2000 --num_batches 500 --batch_size 5000 --disp_V_opt --disp_V_pi --disp_policy --seed

python ./NN_train.py --env Toy-100_zone --beta 0.01 --tau 1.0 --lr 0.5 --dim_emb 100 --num_train 2000 --num_batches 50 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy --seed
python ./NN_train.py --env Toy-100_zone --beta 0.1 --tau 1.0 --lr 0.5 --dim_emb 100 --num_train 2000 --num_batches 50 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy --seed
python ./NN_train.py --env Toy-100_zone --beta 0.5 --tau 1.0 --lr 1.0 --dim_emb 100 --num_train 2000 --num_batches 500 --batch_size 5000 --disp_V_opt --disp_V_pi --disp_policy --seed

# Toy-1000
# Warning: evaluation is very slow.
python ./NN_train.py --env Toy-1000 --beta 0.01 --tau 1.0 --lr 0.001 --dim_emb 1000 --num_train 5000 --freq_eval 50 --thres_eval 0.01 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# CartPole.
python ./NN_train.py --env CartPole --beta 0.01 --tau 1.0 --dim_emb 6 --num_train 2000 --batch_size 10000
python ./NN_online_train.py --env CartPole --beta 0.01 --tau 1.0 --sigma 0.1 --dim_emb 6 --T_train 100000 --batch_size 10000 --freq_update 20 --eps 0.01 --buffer_size 1000000 --off_ratio 0.1 --seed 20

# Pendulum.
python ./NN_train.py --env Pendulum --beta 0.01 --tau 1.0 --num_actions 2 --dim_emb 3 --num_train 2000 --batch_size 10000

# Eval
python ./NN_eval.py --agent_prefix ./log/active/<agent_prefix> --disp_V_opt --disp_V_pi --disp_policy
python ./NN_eval.py --agent_prefix ./log/selected/CartPole_0.0_2000_20_10000_0.5_1.0_20230514_131851 --disp_V_opt --disp_V_pi --disp_policy