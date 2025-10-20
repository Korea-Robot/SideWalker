# Navigation Model 

PPO를 통해서 동적 장애물과 장애물을 잘피하면서 목적지에 잘 도착하는 네비게이션 모델 강화학습 파이프라인.

## backbone

inputs : semantic, rgb, depth 

semantic -> Efficientb0

rgb -> dinov2, segformer : freeze
depth -> perceptnet : freeze

encoder = multiencoder : cross attention

Actor(encoder)
Critic(encoder)

action = [a1,a2] 

a1 : steering in [-1,1]
a2 : throttle in [-1,1] # -1은 후진 1은 전진, 1에 근접한 값으로 이동해야 reward가 잘나옴..


하나 염려하고있는것은 이제 실제 배포할때 target angle이 되기까지 action 분포에서 나온것이랑 달라서 pd제어처럼 넣어줘야할것같은데 미리 학습할때 넣는게 좋을지아닐지 고민이야. 또한, action에서 2개의 output이 나오는데 1번째는 steering이고 두째는 throttle이야. 근데 적어도 0.5 ~1의 값은 줘야 reward가 잘 나오기 때문에 그렇게 exploration을 하도록 초반에 해야하는데 그게 고민이야. throttle이 안나오면 결국 steering도 의미없거든.


# reward 분석 결과

--- Creating Reward Visualizations ---

--- Basic Reward Analysis ---
Reward Component          |     Total Value |       Avg Value |    Triggered Steps
--------------------------------------------------------------------------------
collision_penalty         |         -20.000 |         -10.000 |          2 / 315
env_default_reward        |          67.595 |           0.322 |        210 / 315
goal_proximity            |         303.517 |           2.734 |        111 / 315
speed_reward              |           0.000 |           0.000 |          0 / 315
steering_penalty          |          -3.800 |          -0.050 |         76 / 315
success_reward            |          50.000 |          50.000 |          1 / 315

--- Optimal Navigation Reward Analysis ---
Reward Component          |     Total Value |       Avg Value |    Triggered Steps
--------------------------------------------------------------------------------
checkpoint_progress       |         464.000 |           8.000 |         58 / 315
collision_penalty         |         -40.000 |         -20.000 |          2 / 315
direction_alignment       |         161.597 |           0.513 |        315 / 315
goal_proximity            |           0.000 |           0.000 |          0 / 315
movement_efficiency       |           1.832 |           0.108 |         17 / 315
steering_penalty          |          -7.600 |          -0.100 |         76 / 315
success_reward            |         100.000 |         100.000 |          1 / 315
throttle_smoothness       |         -13.800 |          -0.300 |         46 / 315


Q : 
아래 처럼 reward를 새로 잘 정의해 놓았어. 이대로 움직이기 위해서는 discrete한 action을 내뱉는 categorical distribution action model이 필요할것같아. 그래서 너가 해줄것은 2가지야. 최대한 지금 구조를 유지하면서 단순화시켜서 model.py을 discrete하게 만들어주고, 마찬가지로 구조를 최대한 유지하면서 단순화시켜서 train.py를 calculate_all_rewards함수를 통해 리워드를 적용하도록 해주는것이야.


A : custom reward & discrete action 

    model_discrete.py
    train_discrete.py
    
