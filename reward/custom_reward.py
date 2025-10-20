from metaurban.envs import SidewalkStaticMetaUrbanEnv

class MyEnv(SidewalkStaticMetaUrbanEnv):
    
    def reward_function(*args, **kwargs):
        return -10, {"is_customized": True}
    
env=MyEnv({'object_density': 0.1})
env.reset()
_,r,_,_,info = env.step([0,0])
assert r==-10 and info["is_customized"]
print("reward: {}, `is_customized` in info: {}".format(r, info["is_customized"]))
env.close()
