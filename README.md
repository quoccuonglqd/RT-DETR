
## Quick start

<details open>
<summary>Setup</summary>

```shell
cd rtdetrv2_pytorch
pip install -r requirements.txt
```

The following is the corresponding `torch` and `torchvision` versions.
`rtdetr` | `torch` | `torchvision`
|---|---|---|
| `-` | `2.2` | `0.17` |
| `-` | `2.1` | `0.16` |
| `-` | `2.0` | `0.15` |

</details>


## Usage
<!-- <details> -->
<!-- <summary> details </summary> -->

<!-- <summary>1. Training </summary> -->
<!-- 1. Training -->
<!-- ```shell -->
<!-- CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config --use-amp --seed=0 &> log.txt 2>&1 & -->
<!-- ``` -->

<!-- <summary>2. Testing </summary> -->
<!-- 2. Testing -->
<!-- ```shell -->
<!-- CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only -->
<!-- ``` -->

<!-- <summary>3. Tuning </summary> -->
<!-- 3. Tuning -->
<!-- ```shell -->
<!-- CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint --use-amp --seed=0 &> log.txt 2>&1 & -->
<!-- ``` -->

<!-- <summary>4. Export onnx </summary> -->
<!-- 4. Export onnx -->
<!-- ```shell -->
<!-- python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check -->
<!-- ``` -->

<!-- <summary>5. Inference </summary> -->
<!-- 5. Inference -->

<!-- Support torch, onnxruntime, tensorrt and openvino, see details in *references/deploy* -->
<!-- ```shell -->
<!-- python references/deploy/rtdetrv2_onnx.py --onnx-file=model.onnx --im-file=xxxx -->
<!-- python references/deploy/rtdetrv2_tensorrt.py --trt-file=model.trt --im-file=xxxx -->
<!-- python references/deploy/rtdetrv2_torch.py -c path/to/config -r path/to/checkpoint --im-file=xxx --device=cuda:0 -->
<!-- ``` -->
<!-- </details> -->

6. Train Moe
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -r path/to/last/checkpoint -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml --use-amp --seed=0
```

The argument `-r` is optional and only required when we need to continue training from a checkpoint

The AP score will be logged into the console log as soon as an epoch finishes

In order to change the number of experts, we need to config it in line 27 of file `rtdetrv2_r18vd_120e_coco.yml`
<!-- /home/cuongnq/RT_DETR/rtdetrv2_pytorch/output/rtdetrv2_r18vd_120e_coco_4expert/last.pth -->

7. Test Moe
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -r path/to/checkpoint -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml --test-only
```

8. Inference, calculate speed
```shell
export PYTHONPATH=.
python references/deploy/rtdetrv2_torch.py -r path/to/checkpoint -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml --im-file path/to/image --device=cuda:0
```

<!-- /home/cuongnq/RT_DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -->
<!-- /home/cuongnq/coco2017/val2017/000000515350.jpg -->
<!-- /home/cuongnq/RT_DETR/rtdetrv2_pytorch/output/rtdetrv2_r18vd_120e_coco_4expert/best.pth -->

9. Train Moe + matrix decomposition:
<!-- ```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -t path/to/checkpoint -c path/to/config
``` -->
```
The guideline for training with matrix decomposition will be released soon
```

## Acknowledgments:
This implentation is based mainly on the original implementation of [RT-DETR](https://github.com/lyuwenyu/RT-DETR). We would love to thank the author of RT-DETR for their awesome work
