# A4

## 介绍
antibody_design
传统的计算机辅助的抗体设计流程是通过分子模拟的方法对抗体结构进行建模解析，再用实验的手段去测量活性，这种方法成本高、效率低，难以实现高通量。基于AI辅助的抗体优化，将抗体的突变设计和序列打分筛选交给AI实现，AI模型通过从历史抗体数据库中学习抗体的模式特征，实现快速的抗体序列生成和序列筛选，并且可以使用湿实验反馈的结果优化AI模型，从而实现了高通量、低成本、可交互式的抗体设计。
A4抗体设计模型通过多段功能区域的联合分布改造生成抗体序列，能够实现抗体功能设计、序列嫁接和活性预测等多种任务。


## 安装教程

### 硬件支持情况

| 硬件平台      | 操作系统        | 状态 |
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️ |
|               | Ubuntu-aarch64  | ✔️ |
|               | EulerOS-aarch64 | ✔️ |
|               | CentOS-x86      | ✔️ |
|               | CentOS-aarch64  | ✔️ |

### 软件版本
依赖配套如下表

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件   |                                                               |
| 昇腾 CANN        |                                                               |
| MindSpore        | [MindSpore 1.9.0](https://www.mindspore.cn/install/)         |
| MindSPONGE       | 0.6.0                                                     |
| Python           | >=3.7                                           |

## 可用的模型

| 所属模块      | 文件名       | Model URL|
|------|-----|----|
| cdr生成    | `cdr_design.ckpt`    | [下载链接](https://zenodo.org/records/15510541/files/cdr_design.ckpt?download=1)           |
| fwr嫁接     | `cdr_grafting.ckpt`  | [下载链接](https://zenodo.org/records/15510541/files/cdr_grafting.ckpt?download=1)|
## 使用说明

1.  序列前处理

需要安装`abnumber`(https://abnumber.readthedocs.io/en/latest/)
```bash
conda install -c bioconda abnumber
```

执行前处理脚本：
```bash
python seqs_preprocess.py --input_path "xxx" --output_path "xxx"
```
输入参数说明：

`input_path`：输入fasta文件夹路径

`outpu_path`：输出pkl文件路径

输入fasta规范：
```bash
>test_prot_VH
QVRLVQSGAEVKKPGASVKVSCKASGYTFNTYYIHWVRQAPGRGLEWMGIINPSDGSASYVQNLQGRLSMTIDTSTTTVYMELSSLRSEDTAIYYCARRTLRAFPEWELLVDYWGQGSLVTVSS
>test_prot_VL
QSVLTQPPSASGTPGQRVTFSCSGSTSDIGSNSVYWYQQVPGTAPKLLIYRNNQRPSGVPDRFSGSKSGTAASLAISGLRSEDEADYFCATWTNVPSGRWVFGGGTKLTVL
```
蛋白名+`_VH`表示重链，蛋白名+`_VL`表示轻链


2.  模型推理

(1) 参数说明：

a. `--mas_seqs`最大生成序列数目

b. `--area` 改造区域，可选输入：`cdr3_aa`, `all_cdr`, `all_fwr`, 

对应参数说明：

`cdr3_aa`: 改造CDR3

`all_cdr`: 改造CDR1+CDR2+CDR3

`all_fwr`: 嫁接CDR1+CDR2+CDR3

c. `--design_chain` 改造链，可选输入：`heavy`, `light`, `pair`,

`heavy`: 改造重链

`light`: 改造轻链

`pair`: 同时改造轻重链

d. `numbering`  编号方式，可选输入：`chothia`, `imgt`,

`imgt`: imgt抗体编号方式

`chothia`: chothia抗体编号方式

(2) cdr设计
```bash
python infer_design.py \
    --area=cdr3_aa \
    --design_chain=heavy \
    --numbering=chothia \
    --ckpt_url=/xxx/design.ckpt \
    --pkl_url=/xxx/
```
pkl_url是步骤1前处理完生成的文件路径，ckpt_url是序列设计的checkpoint文件路径

(3) cdr嫁接
```bash
python infer_grafting.py \
    --area=all_fwr \
    --design_chain=pair \
    --numbering=imgt \
    --ckpt_url=/xxx/grafting.ckpt \
    --pkl_url=/xxx/
```
pkl_url是步骤1前处理完生成的文件路径，ckpt_url是序列嫁接的checkpoint文件路径



