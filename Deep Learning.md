# 神经网络算法 （Deep Learning）

## 什么是神经网络

神经网络，通俗来说就是三个步骤
1. 定义一系列“函数”
2. 函数拟合
3. 找到拟合效果最好的函数

### 神经网络中的函数———神经元（Neural Network）

![image](./pics/logistics_regression.png)

每个神经元的输出由其所有输入的非线性函数计算得出，该函数称为激活函数。每个连接处的信号强度由权重（weight）决定，在训练过程中学习的便是这个权重和偏差（bias，相当于常数的权重）。

<img src="./pics/Colored_neural_network.png" width="30%" height="30%">

神经网络便是由不同神经元连接而成，而每个神经元又可以分为输入层，输出层，隐层（也有可能没有隐层）。

<img src="./pics/fully_connect_feedforward_network.png" width="50%" height="50%">

基于不同的方式，便可以得到不同的神经网络层，如全连接层便是前一层的神经元与后一层的神经元两两之间互相连接。
