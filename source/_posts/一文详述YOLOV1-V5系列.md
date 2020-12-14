---
title: 一文详述YOLOV1-V5系列
date: 2020-12-04 13:51:20
index_img: /images/yolo.png
tags:
---

​	`YOLO`系列应该在目标检测中家喻户晓，今天阳光明媚，趁此写一篇长文，也是我的第一篇博客，详细描述一下`YOLO`系列以及里面的细节。（<font color="red">Let's  Go! 😎</font>）

​	首先需要说明的是，从`YOLO V1` 的题目 `You Only Look Once: Unified, Real-Time Object Detection` ，圈下重点，首先是`Unified`,  `YOLO`是一个统一的网络，没有region proposal 的这个过程。（region proposal 主要是`R-CNN`系列中，首先提取的可能的目标区域 ），所以YOLO 是一个`One-Stage` 检测网络。其次，YOLO 将检测统一视为一个**回归**问题，而R-CNN 把检测结果分为两部分，物体类别的分类，即检测出来的物体是属于哪一类（分类问题），物体的具体位置，即Bounding box 的坐标（回归问题）。这是对于`Unified`的理解，其次是`Real-Time`,实时性，也就是YOLO系列的检测性能十分高效，速度很快。

## YOLO V1 

 [论文链接][https://arxiv.org/pdf/1506.02640.pdf]

Yolo V1 将整张图片作为网络的输入，直接在最后的输出层回归 Bounding Box 的位置以及物体所属的类别。

![YOLOV1_model](yolov1model.png)

​	如上图所示，把一幅图像分为 `S×S` 个网格（grid cell）, 如果某个物体的中心落在了这个格子里，那么这个格子就负责预测这个物体。

​	首先给出一个公式，对于每张图像，输出`S×S×(5×B+C)` 这样的一个tensor 。

​	接下来具体理解一下这个公式: 

   `S×S`, 这是一共有这么多个网格；

​	5 个是 （4+1）个构成：

​    `4`个 代表着BBox信息，即`(x, y, w, h)`, x, y 是物体的中心，w,h 是物体的宽度和高度；

  （注意 ：YOLO系列的数据形式都是x y w h, 即中心点坐标和宽高； R-CNN系列的则是x y x y，即左上角和右下角的坐标。）

​	`	1`  个代表的是置信度，confidence计算方法: $\operatorname{Pr}($ Object $) * \mathrm{IOU}_{\text {pred }}^{\text {truth }}$,  第一项指的是物体是否落在该网格里，如果没有取0，如果在，取1；第二项指的是预测的 BBox 和 Ground-truth的 IoU 大小。两者相乘就是 confidence. IoU 指的是两个矩形框的交并比。

在这里，放一个计算 IoU 的代码（听说面试常考）, 方便理解：

```python
def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
```

B 指的是一个网格预测 B 个BBox； C指的是物体的类别；

举个例子，论文里对于YOLO在PASCAL VOC中 的实验，S = 7, B = 2, C = 20,  因此就是输出7 × 7 × 30 的tensor。

![YOLOV1_model](yolov1framework.png)

**注意:**  这里B = 2 是一个可调节的 参数，如果B = 2， 每个网格产生两个预测框，最后只选定置信度较大的作为输出，也就是说，最终每个方格只输出一个预测框，所以每个方格只能预测一个物体，如果出现多个物体的中心落在了一个小网格里面，那么该网格也是只能预测一个物体，这种情况会出现漏检。换句话说，对于一个输入图像，只能预测出 S × S 个物体。（当然，S也是可以调节的。） 

​	在测试的时候，每个网络预测的class概率和BBox的confidence概率相乘，得到该BBox的 class-specific confidence score

计算公式：
$$
\operatorname{Pr}\left(\text { Class }_{i} \mid \text { Object }\right) * \operatorname{Pr}(\text { Object }) * \mathrm{IOU}_{\text {pred }}^{\text {truth }}=\operatorname{Pr}\left(\text { Class }_{i}\right) * \mathrm{IOU}_{\text {pred }}^{\text {truth }}
$$
​	接着是 **非极大值抑制(NMS)**, NMS 用来去除冗余的重叠框，

![NMS](nms.png)

NMS算法流程：

   给出一张图片和上面许多物体检测的候选框（即每个框可能都代表某种物体），但是这些框很可能有互相重叠的部分，我们要做的就是只保留最优的框。假设有`N`个框，每个框被分类器计算得到的分数为`Si, 1<=i<=N`

1. 建造一个存放待处理候选框的集合H，初始化为包含全部N个框；建造一个存放最优框的集合M，初始化为空集
2. 将所有集合 H 中的框进行排序，选出分数最高的框 m，从集合 H 移到集合 M
3. 遍历集合 H 中的框，分别与框m计算交并比（IoU），如果高于某个阈值（一般为0.3~0.5），则认为此框与m重叠，将此框从集合 H 中去除.
4. 回到第2步进行迭代，直到集合 H 为空。集合 M 中的框为我们所需。

为便于理解，放一个用 `numpy` 写的NMS代码： 

```python
import numpy as np

def NMS(dets, thresh):

    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]


    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp


if __name__ == "__main__":
    dets = np.array([[310, 30, 420, 50, 0.6],
                     [20, 20, 240, 210, 1],
                     [70, 50, 260, 220, 0.8],
                     [400, 280, 560, 360, 0.7]])
    # 设置阈值
    thresh = 0.4
    keep_dets = NMS(dets, thresh)
    # 打印留下的框的索引
    print(keep_dets)
    # 打印留下的框的信息
    print(dets[keep_dets])
```

  最后我们看一下YOLO V1 中训练的**损失函数**:

![loss](yolov1loss.png)

我们深入探讨理解一下这个看起来很复杂的代价函数：

1. 首先，第一行是预测框中心点的约束，$\mathbb{1}_{i j}^{\mathrm{obj}}$  是一个控制函数，标签中包含物体的那些网格，值为1，否则为 0，只对真正有物体的网格做损失计算。
2. 第二行是预测框的宽高的约束，我们注意看 这里多了个根号，为什么呢？ 这是比较巧妙的一种做法，旨在消除大尺寸和小尺寸的框之间的差异，比如同样是10个像素的误差，对于几百乘几百的大物体框，几乎没有影响，但是对于本身就只有几十像素的框，影响则非常大。因此如果不开根号，训练过程往往会只优化调整那些尺寸比较大的框。
3. 第三行和第四行都是对预测框置信度的约束， 当该网格中含有物体的时候，置信度的 gt 为 预测框与真实框的 IoU, 没有物体时，置信度的 gt 为0。
4. 第五行 为 物体类别 的约束，对应的类别位置gt为1，其余位置为0。

这里有两个系数，$$\lambda_{coord}$$   $$\lambda_{noobj}$$，前者论文中为5，后者论文中设为0.5，

 因为物体检测其实是类别不均衡的问题，只有少数网格有物体，因此如果不进行权重约束，网络会倾向于约束不含有物体的格点，但我们更想要的是约束含有物体的网格预测的更准确。因此采取这样一个权重约束。

**训练技巧：** x, y 不是直接回归中心点坐标的数值，而是回归相对于网格点左上角坐标的位移值。

<font color="blue">**读到这里，相信你对YOLO V1 应该有一个较为全面详细的了解了吧~~**🤭</font>

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

## YOLO V2

[论文链接][https://arxiv.org/pdf/1612.08242.pdf]

了解完YOLO V1，咱们来看看小马哥的YOLO V2，（至于为什么叫小🐎哥，去YOLO官网看看大佬的简历就知道了哈哈哈）

​		前面说到YOLO V1 的 `unified`,  `real-time`， 但也有不足之处，就是因为漏检呢，召回率会比较低。YOLO V2 对 YOLO V1进行了很多改进。先说说论文题目怎么叫 YOLO9000， YOLO 9000 是采用了和 YOLO V2 一样的模型， 在COCO 和 ImageNet 上一起训练，使得训练后的模型可以实现多达9000种物体的实时检测。

**YOLO V2 的改进之处:**

先看看论文里这个从YOLO V1 到 YOLO V2的路线图：

![the path from YOLO to YOLO V2](yolov2path.png)

**① Batch norm:**  

​		这里先简单介绍一些Batch norm, 后面会有文章专门介绍各种 norm; 

​		CNN在训练过程中网络每层输入的分布一直在改变, 会使训练过程难度加大，但可以通过normalize每层的输入解决这个问题，具体做法是: 数据的输入值减去其均值然后除以数据的标准差，然后引入一个平移变量和缩放变量，再计算归一化的值。

通过 BN，我们可以看到 mAP 得到了`2.2` 个点的提高。

**②  High Resolution Classifier:** 

​		几乎所有的检测器都会使用在 ImageNet 上预训练的分类器，YOLO V1 中用`224 × 224` 的图片来预训练分类网络，检测时增大分辨率至 `448 × 448`，这意味着网络不得不同时学习检测任务和适应新的更大的分辨率。而在YOLO V2 中，作者先`224 × 224`预训练，然后用`448 × 448`的分辨率在 ImageNet 上finetune ，最后作者再对检测网络部分（也就是后半部分）也进行fine tune， 这个操作可以提升大约`4`个点。

**③ Convolutional With Anchor Boxes：**

​		对于Anchor Box 可以理解为预定义的一些边框，在YOLO 中，作者是通过K-Means 聚类来得到这些边框，我们先看一组数:

```
anchors = 0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
```

咦？😯 怎么最大的才9点多？ 这是相对于最后一个Feature map 大小，YOLO V2 检测时候输入大小为`416 × 416`， 降采样32倍，变成`13 × 13`。这组数一共十个，每相邻两个一组，因此是五个anchor boxes，为什么是五个呢？这是因为作者权衡了一下，在召回率和模型复杂度之间取了一个trade -off，原文有这个比较图。

​		K-means的方法：作者并没有采用欧氏距离，而是采用了`d(box,centroid)=1-IOU(box,centroid)`，这样一来，便和框的尺寸大小无关，而是希望得到更高的IOU.

​		采用了Anchor Boxes机制之后，我们分析一下，YOLOV1中，只能预测98 (`7*7*2`)个框，而在V2 中，可以预测出`13*13*num_anchors`个框，这使得YOLO V2的召回率大大提升。

**④ Direct location prediction：**

​		先看一张示意图：

![Bounding boxes with dimension priors and location prediction ](yolov2location.png)

​		YOLO V2 也是预测边界框中心点相对于对应cell左上角位置的相对偏移值(offset)，为了将边界框中心点约束在当前cell中，使用sigmoid函数处理偏移值，这样偏移值便控制在`(0,1)`之间。对于每一个BBox, 网络预测五个数，$t_x,t_y,t_w,t_h,t_o$, 图中$c_x,c_y$指的是当前cell左上角坐标，$p_w,p_h$指的是先验框的宽度和高度。这样，BBox的位置和大小便可以由图中的公式计算出来，

**⑤ Fine-Grained Features：**

​		最后的feature map 大小为`13×13`，作者为了加入更细粒度的特征，引入了叫`passthrough` 层， 把前面层分辨率为`26×26`的特征图和这个`13×13`的特征图`concat`起来， 具体做法是: 首先把`26×26×512-->13×13×2048`，然后在channel维度上面 concat.

**⑥ Multi-scale Training:**

​		这是作者采用的多尺度训练策略，这个trick应该不管在分割检测分类任务都通用把。在YOLO V2 中，作者是每十个batch就更换一次尺寸大小，由于YOLO 是降采样32倍，因此所谓的多尺度的尺寸也都是32的倍数，比如论文里用的`{320,352,...,608}`, 多尺度训练使得模型的鲁棒性得到了提升。从表中可以得知，这一个trick会得到约`1.4`个点的提升。

接下来，我们来看看YOLO V2 的loss函数，深入理解一下是怎么训练的：

![loss](yolov2loss.png)

​		首先我们需要明白：对于训练图片中的ground truth，和 V1一样若其中心点落在某个cell内，那么该 cell 内的5个 anchors 所对应的边界框负责预测它。和 V1不同的是，因为 V2 有了5个不同的先验框，我们不再和网络输出的预测框计算IOU值，而是用**先验框与ground truth计算IOU值**，（这里计算IOU的时候，不考虑坐标，只考虑形状，所以先将先验框与ground truth的中心点都偏移到原点，然后计算出对应的IOU值，最大的先验框对应的预测框ground truth匹配，而剩余的4个先验框对应的预测框不与该ground truth匹配。

然后应该明确哪些预测框需要计算损失。首先，对于网络输出的所有预测框，每个预测框都与各个ground truth计算IOU值，在这些IOU值中取最大值就是Max IOU。然后根据Max IOU和是否匹配ground truth可以把预测框分为三类并计算损失。

```
1.如果某一个预测框的Max IOU小于0.6(说明它预测背景)且不匹配ground truth, 则这个预测框计算置信度损失。
2.如果某一个预测框的Max IOU大于0.6，但是不和ground truth匹配，我们忽略其损失。
3.如果某一个预测框匹配ground truth。那么我们就要对这个预测框计算坐标、置信度和类别概率损失。
```

​		关于第二点，因为YOLO V2的一个ground truth只对应一个先验框,对于那些没有与ground truth匹配的先验框（与预测框对应），尽管Max_IOU高于阈值也要忽略，不计算任何误差。

​		 1. 第一行loss，这部分主要约束的就是没有物体的框，

​	`1MaxIOU<Thresh` 实际表述最后会得到一个类似于 `[13, 13, 5, 1]`的一个`bool`张量，这个`MaxIOU<Thresh`针对的是：
​		① annotation file里面所有的boxes
​		② 预测的[13, 13, 5, 4] 的 `[tx, ty, tw, th]`，记作coorT
​		这里将所有的boxes和每一个coorT计算得到一个iou列表，然后取到这个iou列表的最大值，如果这个最大值小于阈值（常见0.6），则将这个no object框计入到loss中，因为这部分被认定为没有物体的框。
​		由于计算出来 feature map 之后，我们主要度量的`objectness=1`的部分（即有物体部分）很少，绝大多数为没有物体的部分，需要将没有物体部分（no object）的部分考虑进来，避免最后模型不能分辨前景和背景。

​		2. 第二行loss，这部分主要只在前12800次迭代中采用，用于促进网络学习到anchor的形状。

​		3. 第三行loss， 这部分是坐标的损失，预测框和真实框的坐标误差计算

​		4. 第四行loss， 置信度损失，在计算的时候是用的预测框和gt的IOU计算

​		5. 第五行loss， 类别的误差计算。

<font color="blue">**读到这里，相信你对YOLO V2 应该有一个较为全面详细的了解了吧~~**🤭</font>

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

## YOLO V3

[论文链接][https://pjreddie.com/media/files/papers/YOLOv3.pdf]

​		YOLOV3 的题目 `YOLOv3: An Incremental Improvement`，看看这题目取得，可能也只有小马哥这种大佬敢这么随心所欲吧！

论文写的非常随意，但通过前面YOLO V1和 V2的理解，有助于我们理解V3。

​		首先我们来看一张结构图，这张图是从`Levio`这里拿过来的：

![yolov3](yolov3.png)

​		我们先来看一下这个框架图：

​		**DBL：** 这个就是卷积+BN+Leakly Relu

​	（关于leakly relu, 看下面这个图就一目了然了~ 是一个relu进化的激活函数）

![activation functions](relu.png)

​		**Resn：**这里的`n`指的是这个res block里面含有几个res_unit，因为有了残差结构，可以使得网络变得更深，（从YOLO V2的 `darknet-19` 进化到 `darknet-53`）

我们可以看到：YOLOV3里面是**没有** `Pooling` 和 `Fc` 的，通过步长为2的卷积来降采样，这里 V3 和 V2一样，都是降采样32倍。

主要改进:

​		看网络的结尾，是一个多尺度的预测，输出了三个不同尺度的feature map，这可以增加网络的鲁棒性，越精细的grid cell可以检测出越精细的物体，具体做法就是类似 `coarse-to-fine` 的思想，将feature map上采样后和前面的feature map在通道维度上`concat`.

​		