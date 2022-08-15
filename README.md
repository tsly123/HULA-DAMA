## DAMA - Student Collaboration Improves Self-Supervised Learning: Dual-Loss Adaptive Masked Autoencoder for Brain Cell Image Analysis

This is a PyTorch/GPU implementation of the paper Student Collaboration Improves Self-Supervised Learning: Dual-Loss Adaptive Masked Autoencoder for Brain Cell Image Analysis

* This repo is based on PyTorch=1.10.1 and timm=0.5.4

Below is the fine-tune result of DAMA compared to other state-of-the-art methods pretrained on **brain cells dataset** and **ImageNet-1k**. Please see the paper for detailed results.

### Brain Cell datasets
Manually collected set *Aug-30k* and noisy set *Real-30k*
|                    | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | Avg. &#8593;          | Err. &#8595; |
|--------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------------------------:|:-------------------:|
| Random   init.     | 91.75 | 91.19 | 92.75 | 92.69 | 92.56 | 92.31 | 91.44 | 91.06 | 93    | 91.06 | 91.98(+0.00)             | 8.02              |
| Data2vec 800 (4h)  | 90.56 | 90.25 | 91.75 | 92.31 | 91.94 | 92.62 | 91    | 91.38 | 92.5  | 90.88 | 91.59(-0.39)             | 8.41              |
| MOCO-v3 500 (6h)   | 90.94 | 91.5  | 92.38 | 92.38 | 92.56 | 92.12 | 91.25 | 90.94 | 92.69 | 90.75 | 91.75(-0.23)             | 8.25              |
| MAE 800 (4h)       | 94.69 | 93.81 | 95.19 | 95.25 | 95    | 93.56 | 94.62 | 93.88 | 95.44 | 94    | 94.54(+2.56)             | 5.46              |
| DAMA-rand 500 (3h) | 94.69 | 94.19 | 94.81 | 95.81 | 94.50 | 94.00 | 94.88 | 94.69 | 95.25 | 94.81 | 94.76(+2.78) | 5.24  |
| DAMA 500 (5h)      | 95.5  | 94.5  | 95.69 | 96.25 | 95.56 | 95.44 | 95.62 | 94.94 | 95.69 | 95.25 | ***95.47(+3.49)***      | ***4.53***     |


### Pretrained on ImageNet-1k with ViT-Base
Due to computational resource, DAMA is trained **only once** without any ablation experiment for ImageNet and with similar configuration as for trained the brain cell dataset.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
    <th>Methods</th>
    <th>Pretrained Epochs</th>
    <th>Acc</th>
</tr>
<!-- TABLE BODY -->
<tr>
<td align="left">Moco-v3</td>
<td align="center">600</td>
<td align="center">83.2</td>
</tr>
<tr>
<td align="left">BEiT</td>
<td align="center">800</td>
<td align="center">83.4</td>
</tr>
<tr>
<td align="left">SimMIM</td>
<td align="center">800</td>
<td align="center">83.8</td>
</tr>
<tr>
<td align="left">Data2Vec</td>
<td align="center">800</td>
<td align="center">84.2</td>
</tr>
<tr>
<td align="left">DINO</td>
<td align="center">1600</td>
<td align="center">83.6</td>
</tr>
<tr>
<td align="left">iBOT</td>
<td align="center">1600</td>
<td align="center">84.0</td>
</tr>
<tr>
<td align="left">MAE</td>
<td align="center">1600</td>
<td align="center">83.6</td>
</tr>
<tr>
<td align="left">DAMA</td>
<td align="center">500</td>
<td align="center">83.17</td>
</tr>    
</tbody></table>

### Pre-training DAMA
```
python submitit_pretrain.py --arch main_vit_tiny \
      --batch_size 64 --epochs 500 --warmup_epochs 40 \
      --mask_ratio 0.8 --mask_overlap_ratio 0.5 --last_k_blocks 6 --norm_pix_loss \
      --data_path path_to_dataset_folder \
      --job_dir path_to_output_folder \
      --nodes 1 --ngpus 4
```

### Fine-tuning DAMA
```
python submitit_finetune.py --arch main_vit_tiny \
      --batch_size 128 --epochs 150  \
      --data_path path_to_dataset_folder \
      --finetune path_to_pretrained_file \
      --job_dir path_to_output_finetune_folder \
      --dist_eval --nodes 1 --ngpus 4
```
