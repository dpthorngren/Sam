import numpy as np
import sam

# Griddy tests
x = np.sin(np.linspace(0,np.pi/2,100))
z = x*np.sin(x[:,np.newaxis])+4*np.log(x[np.newaxis,:]+1)
f = sam.Griddy((x,x),z)
# print f((.5,np.random.rand(1000)))
print f((.363,.634))
points = np.array((.95,.9))
displacement = np.array((-.15,.2))
bounced = np.ones(2,dtype=np.intc)
print '====='
print points
f.bounceMove(points,displacement,bounced)
print points
print bounced
# sam.test()

# f = sam.Sam(2,.01)
# print f.run(100,100,np.random.rand(2))
# sam.test()
