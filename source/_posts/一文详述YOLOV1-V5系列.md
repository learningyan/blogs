---
title: 一文详述YOLOV1-V5系列
date: 2020-12-04 13:51:20
index_img: /images/yolo.png
tags:
---

​	`YOLO`系列应该在目标检测中家喻户晓，今天阳光明媚，趁此写一篇长文，也是我的第一篇博客，详细描述一下`YOLO`系列以及里面的细节。 Let's Go!

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
​	最后是 非极大值抑制(NMS), NMS 用来去除冗余的重叠框，

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
    dets = np.array([[310, 30, 420, 5, 0.6],
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

