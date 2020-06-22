# CQL
 Implementation of CQL in "Conservative Q-Learning for Offline Reinforcement Learning" based on BRAC family.
 
**Usage**:
 Plug this file into BRAC architecture and run train_offline.py. 
 
 ```
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
    --alsologtostderr --sub_dir=cql+$VARIANT+$PLR+$ALPHA \
    --root_dir=$ROOT_DIR \
    --env_name=$ENV \
    --agent_name=cql \
    --data_root_dir='data' \
    --data_name=$DATA \
    --total_train_steps=$TRAIN_STEPS \
    --seed=$SEED \
    --gin_bindings="cql_agent.Agent.variant='p'" \
    --gin_bindings="cql_agent.Agent.train_alpha=False" \
    --gin_bindings="cql_agent.Agent.alpha=1.0" \
    --gin_bindings="cql_agent.Agent.train_alpha_entropy=False" \
    --gin_bindings="cql_agent.Agent.alpha_entropy=0.0" \
    --gin_bindings="train_eval_offline.model_params=(((300, 300), (200, 200), (750, 750)), 2)" \
    --gin_bindings="train_eval_offline.batch_size=256" \
    --gin_bindings="train_eval_offline.seed=$SEED" \
    --gin_bindings="train_eval_offline.weight_decays=[$L2_WEIGHT]" \
    --gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3), ('adam', 1e-3))" &
```
