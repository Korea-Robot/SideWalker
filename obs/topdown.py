from metaurban import TopDownMetaUrban

env = TopDownMetaUrban({'object_density': 0.1})
try:
    o,i = env.reset()
    for s in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        env.render(mode="top_down")
        if tm or tc:
            break
            env.reset()
finally:
    env.close()
