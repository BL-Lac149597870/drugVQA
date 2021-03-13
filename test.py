'''
Author: QHGG
Date: 2021-02-27 20:53:09
LastEditTime: 2021-02-27 22:23:32
LastEditors: QHGG
Description: 
FilePath: /drugVQA/test.py
'''
import torch
print(torch.__version__) # 1.2.0+cpu
# 默认创建requires_grad = False 的Tensor
x1 = torch.ones(1, requires_grad=True)  # create a tensor with requires_grad=False (default)
x2 = torch.ones(1, requires_grad=True)  # create a tensor with requires_grad=False (default)
w1 = torch.tensor(2.0)
print(x1.requires_grad) # False
x2 = x1 + w1
x1 = x1 + x2
x2 = x1 + w1
x2.backward()

exit()
y = torch.ones(1)
z = x + y
# 因为两个Tensor x,y，requires_grad=False.都无法实现自动微分，
# 所以操作（operation）z=x+y后的z也是无法自动微分，requires_grad=False
print(z.requires_grad) # False

# 可以验证， 因而无法autograd，执行下面程序z.backward() 报错
# print(z.backward())
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

w = torch.ones(1, requires_grad=True)
print(w.requires_grad) # True

# 因为total的操作中输入Tensor w的requires_grad=True，因而操作可以进行反向传播和自动求导。
total = w + z
print(total.requires_grad) # True

total.backward()
print(w.grad) # tensor([ 1.])
print(total.grad)
#total.data -->  <class 'torch.Tensor'> tensor([3.])
print("total.data --> ",type(total.data), total.data) 

# 由于z，x，y的requires_grad=False,所以并没有计算三者的梯度
print(z.grad, x.grad, y.grad) # None None None
