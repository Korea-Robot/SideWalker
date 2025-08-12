from metaurban import TopDownMetaUrban

import matplotlib.pyplot as plt


env = TopDownMetaUrban({'object_density': 0.1})
try:
    o,i = env.reset()
    for s in range(1, 100000):
        obs, reward, tm, tc, info = env.step([1, 0.5])
        env.render(mode="top_down")
        plt.imshow(obs[:,:,0], cmap='gray')
        plt.show()
        if tm or tc:
            break
            env.reset()
finally:
    env.close()
