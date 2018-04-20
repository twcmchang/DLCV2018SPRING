
## 1. Train

```
python3 train --init_from <pretrained npy> --save_dir <model checkpoint dir> --mode <FCN8s, FCN16s, or FCN32s>
```
- mode = FCN32s
- batch size = 32
- num of epoch = 135
- learning rate is 1e-4 and half per 2000 steps
- without dropout and weight decay
- input dimension = (256, 256, 3)
- output dimension = (256, 256, 3) and then resize to (512, 512, 3)

Baseline FCN32s model's mIoU = 0.656


## 2. Test

```
python3 test.py --test_dir <testing data dir> --save_dir <model checkpoint dir> --plot_dir <plot dir>
```

## 3. Evaluate (mIoU), provided by TAs

```
python3 mean_iou_evaluate.py -g <ground truth dir> -p <plot dir>
```
