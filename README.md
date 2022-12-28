[PyTorch] GhostNet: More Features from Cheap Operations
=====
PyTorch implementation of "GhostNet: More Features from Cheap Operations"

## Concept
<div align="center">
  <img src="./figures/ghost_module.png" width="400"><img src="./figures/ghost_bottleneck.png" width="400">    
  <p>Concept ot the GhostNet [1].</p>
</div>

## Results

### Loss & Accuracy

<img src="./figures/GhostNet_opt.svg" width="350"><img src="./figures/GhostNet_acc.svg" width="350">    

### Performance

|Indicator|Value|
|:---|:---:|
|Accuracy|0.99230|
|Precision|0.99233|
|Recall|0.99221|
|F1-Score|0.99225|

```
Confusion Matrix
[[ 976    0    0    0    0    1    3    0    0    0]
 [   0 1135    0    0    0    0    0    0    0    0]
 [   2    1 1027    0    0    0    0    2    0    0]
 [   0    1    0  998    0   10    0    0    1    0]
 [   0    0    1    0  977    0    0    1    0    3]
 [   1    0    0    3    0  887    1    0    0    0]
 [   3    6    1    0    0    1  947    0    0    0]
 [   2    8    0    1    2    0    0 1015    0    0]
 [   2    1    2    1    1    5    0    0  961    1]
 [   3    1    0    1    4    0    0    0    0 1000]]
Class-0 | Precision: 0.98686, Recall: 0.99592, F1-Score: 0.99137
Class-1 | Precision: 0.98439, Recall: 1.00000, F1-Score: 0.99213
Class-2 | Precision: 0.99612, Recall: 0.99516, F1-Score: 0.99564
Class-3 | Precision: 0.99402, Recall: 0.98812, F1-Score: 0.99106
Class-4 | Precision: 0.99289, Recall: 0.99491, F1-Score: 0.99390
Class-5 | Precision: 0.98119, Recall: 0.99439, F1-Score: 0.98775
Class-6 | Precision: 0.99579, Recall: 0.98852, F1-Score: 0.99214
Class-7 | Precision: 0.99705, Recall: 0.98735, F1-Score: 0.99218
Class-8 | Precision: 0.99896, Recall: 0.98665, F1-Score: 0.99277
Class-9 | Precision: 0.99602, Recall: 0.99108, F1-Score: 0.99354

Total | Accuracy: 0.99230, Precision: 0.99233, Recall: 0.99221, F1-Score: 0.99225
```

## Requirements
* PyTorch 1.11.0

## Reference
[1] Han, Kai, et al. <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html">"Ghostnet: More features from cheap operations."</a> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
