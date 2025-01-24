import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#定义线性回归模型
class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(3, 1)
    def execute (self,x) :
        x = self.layer1(x)
        return x
 
def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 2)   #随机生成数据
        x = np.hstack((x, np.ones((batch_size, 1))))    #增加一个用于模拟b
        w = np.array([1, 2, 3]) #自定义参数w1,w2,b
        y = np.dot(x, w) + np.random.normal(0, 0.1, batch_size) #随机生成数据得到y+高斯噪声
        yield jt.float32(x), jt.float32(y)

np.random.seed(0)
jt.set_seed(30)
n = 3000
batch_size = 50

model = Model()
learning_rate = 0.0001
optim = nn.SGD(model.parameters(), learning_rate)
'''
#打印训练前的参数(w1,w2,b)
x = np.array([1,0,0])
x = jt.float32(x)
print(model(x))
x = np.array([0,1,0])
x = jt.float32(x)
print(model(x))
x = np.array([0,0,1])
x = jt.float32(x)
print(model(x))
'''
loss_list = []
#训练
for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    loss = jt.sqr(pred_y - y)
    loss_mean = loss.mean()
    optim.step (loss_mean)
    if i % 10 == 0:
        loss_list.append(loss_mean.numpy().sum())

#训练完的参数w1,w2,b
'''
x = np.array([1,0,0])
x = jt.float32(x)
print(model(x))
x = np.array([0,1,0])
x = jt.float32(x)
print(model(x))
x = np.array([0,0,1])
x = jt.float32(x)
print(model(x))
'''
#画Loss
x = np.linspace(10,3000,300)
plt.plot(x, loss_list)
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.show()

#画数据散点
fig = plt.figure(figsize=(20,10))
axl3 = fig.add_subplot(projection='3d')
x_l = []
y_l = []
z_l = []
for i,(x,y) in enumerate(get_data(n)):
    for x_d, y_d, o in x:
        x_l.append(x_d)
        y_l.append(y_d)
    for z_d in y:
        z_l.append(z_d)
x_l = np.array(x_l[:1500])
y_l = np.array(y_l[:1500])
z_l = np.array(z_l[:1500])
axl3.scatter3D(x_l,y_l,z_l,cmap=plt.cm.winter)

#画超平面
x = np.linspace(0,1,500)
y = np.linspace(0,1,500)
x,y = np.meshgrid(x,y)
z_pred = float(model(jt.float32(np.array([1,0,0])))) * x + float(model(jt.float32(np.array([0,1,0])))) * y + float(model(jt.float32(np.array([0,0,1]))))
axl3.plot_surface(x,y,z_pred,cmap=plt.cm.winter)
axl3.set_xlabel('x', fontsize=15)
axl3.set_ylabel('y', fontsize=15)
axl3.set_zlabel('z', fontsize=15)
plt.show()