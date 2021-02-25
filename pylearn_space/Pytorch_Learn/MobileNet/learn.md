## MobileNet知识点
### 2017年提出MobileNetv1
亮点:
-   Depthwise Convolution(DW卷积)
-   增加超参数$\alpha,\beta$  

$\alpha$:卷积核个数的倍率,控制卷积核的个数。  
$\beta$:分辨率超参

**对比传统卷积与DW卷积：**  

传统卷积：卷积核channel=输入特征卷积channel,输出特征矩阵channel=卷积核个数  

DW卷积：卷积核channel=1,输入特征矩阵channel=卷积核个数=输出特征矩阵channel。
> 经过DW矩阵后，特征矩阵的深度不会发生变化  
  
**Depthwise Separable Conv(深度可分神经网络)**

由两部分组成：DW卷积核PW卷积【PW卷积：卷积核为1的普通卷积】

参考：<https://yinguobing.com/separable-convolution/>

缺点：DW卷积部分卷积核会废掉，即等于0。

pytorch中分组卷积的实现：  
>直接调用torch.nn.Conv2d()就可以实现分组卷积。在Conv2d里面有一个参数叫groups，这个参数是实现这一功能的关键。groups默认为1，也就是说将输入分为一组，此时是常规卷积。groups数值为几表示的就是将输入通道分为几组。当groups=in_channels的时候，表示的就是将输入的每一个通道都作为一组，然后分别对其进行卷积，输出通道数为k，最后再将每组输出串联，最后通道数为in_channels*k

---
### 18年提出的MobileNetv2

亮点:
-   Inverted Residual(倒残差结构)
-   Linear Bottlenecks  

Inverted Residual block：  
1.  $\color{blue}{1×1卷积升维}$
2.  $\color{blue}{3×3卷积DW}$
3.  $\color{blue}{1×1卷积降维}$
 
 $y=ReLU6(x)=min(max(x,0),6)$

实验得出ReLU激活函数对低维特征信息造成大量的损失。倒残差输出的低维矩阵，所以使用线性激活代替ReLU减少信息损失。

![网络结构](https://pic.downk.cc/item/5fae453d1cd1bbb86b96276d.jpg)

其中t是扩展因子,tk值的是卷积核的个数；  
c是输出特征矩阵深度channel  
n是bottleneck的重复次数  
s是步距。
  
  
### MobileNetv3

创新点：
- Block(bneck)：基于倒残差结构进行简单的改动,加入注意力机制并更新了激活函数
- NAS搜索
- 重新设计耗时层结构（第一个卷积层的卷积核个数32->16）

> 激活函数优化 sigmoid->h-sigmoid 、 swish->h-swish  
>  
> $h-sigmoid = \frac{ReLU6(x+3)}{6}$
> 
> $h-swish[x] = x\frac{ReLU6(x+3)}{6}$

**注意力机制:**  

对于每一个channel先进行池化处理，再通过两个全连接层得到输出的向量。  

**网络结构：**  

![网络结构](https://img.imgdb.cn/item/6034bcaa5f4313ce253f7d7d.jpg)
