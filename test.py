# import pyximport
import numpy as np
# pyximport.install(setup_args={"include_dirs":np.get_include()})
import hmc
# gd.test()

# x = np.sin(np.linspace(0,np.pi/2,100))
# z = x*np.sin(x[:,np.newaxis])+4*np.log(x[np.newaxis,:]+1)
# f = gd.Griddy((x,x),z)
# f((.5,np.random.rand(1000)))

# f = hmc.HMCSampler(2,.01)
# print f.run(100,100,np.random.rand(2))
hmc.test()
