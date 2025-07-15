# base
这个数据集少额外**中线Fpz**和**枕中线Oz**的数据。

# code部分

## [train.py]
调用了load_references，这个函数默认只读取从idx开始的100个记录。


## [wettbewerb.py]
明确说明不可以修改。如果使用12个通道的对比，需要自己写处理函数。