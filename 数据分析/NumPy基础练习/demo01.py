import numpy as np
import matplotlib.pyplot as plot
x=np.arange(0,6,0.1)
y=np.sin(x)
# 画出图形
plot.plot(x,y)
plot.xlabel("x")
plot.xlabel("y")
plot.title("y=sin ( x )")
plot.show()
# 定义导函数
y1=np.cos(x)
# 画出图形
plot.plot(x,y1)
plot.xlabel("x")
plot.xlabel("y1")
plot.title("y1= cos( x )")
plot.show()
# 画出组合图形
plot.plot(x,y,label='y=sin(x)')
plot.plot(x,y1,label='y=cos(x)',linestyle='--')
plot.xlabel("x")
plot.xlabel("y")
plot.title("y1= sin( x )")
plot.legend()
plot.show()