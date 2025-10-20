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