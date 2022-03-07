## Cell Transformer

The Code for "基于改进Vision Transformer的血细胞图像识别方法研究"

This repository is based on [mmclassification](https://github.com/open-mmlab/mmclassification/).

![architecture](https://s2.loli.net/2022/02/25/i1zLKmHflNMp3kw.png)

## Installation

Please refer to [install.md](https://mmclassification.readthedocs.io/en/latest/install.html) for installation and dataset preparation or use the following command

```
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install -e .
```
## TMAMD Dataset
The datasets are available in this  [Link](https://pan.baidu.com/s/1CnUzgsPqKLIsz5TB2WSZcw) extract code: w52v
## Performance on TMAMD

| Backbone              |  Accuracy Top-1 (%)  |                                   Model                                    |
| :-------------------: | :---------------: |:--------------------------------------------------------------------------:|
| Vision Transfomer Cell | 91.88     | [Link](https://pan.baidu.com/s/1NUOFaVCriodqvzlRIh7exw)  extract code:js7z |

## Get Started

Once the installation is done, you can follow the below steps to test or train the model.

### A quick demo:

```shell
python tools/train.py configs/vision_transformer/vit-base-cell-p16_pt-64xb64_in1k-224.py --work-dir work_dir/vision_transformer_cell/
```
### Config Files:

|               Config File               | Folder                      |    description     |
| :-------------------------------------: | --------------------------- | :----------------: |
|       vision_transformer_cell.py        | mmcls/models/backbones/     |      backbone      |
| vit-base-cell-p16_pt-64xb64_in1k-224.py | configs/vision_transformer/ |  backbone config   |
|     vision_transformer_head_cell.py     | mmcls/models/heads/         | head and loss func |
|         analyze_results_cell.py         | tools/analysis_tools/       |   analyze result   |
|           show_attn_custom.py           | tools/analysis_tools/       | draw attention map |
|              show_tsne.py               | tools/analysis_tools/       |   draw tsne map    |

### Test on TMAMD Dataset

```shell
# single-gpu testing
python tools/test ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} $GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Arguments:
- `CONFIG_FILE`: Path to one of the file in `configs/vision_transformer/vit-base-cell-p16_pt-64xb64_in1k-224.py`.
- `CHECKPOINT_FILE`: Path to the checkpoints.

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Recommended values are: `bbox`, `segm`.
- `--show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing. Please make sure that GUI is available in your environment, otherwise you may encounter the error like `cannot connect to X server`.


## Citation

Please consider citing our paper in your publications if the project helps your research.

```shell
wait for the paper being received
```

