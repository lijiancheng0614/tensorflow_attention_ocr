# TensorFlow Attention OCR

A TensorFlow model for image text extraction problems.

Forked from https://github.com/tensorflow/models/tree/master/research/attention_ocr


## Requirements

1. Install the [TensorFlow library](https://www.tensorflow.org/install/).

2. Data. For example,

    ```
    datasets
    +-- data
    |   +-- fsns
    |       +-- charset_size=134.txt
    |       +-- fsns-00000-of-00001
    +-- fsns.py
    ```


## Train

To train from scratch:

```bash
python train.py \
    --train_log_dir=train_logs
```

To train a model using pre-trained Inception weights as initialization:

```bash
python train.py \
    --checkpoint_inception=inception_v3.ckpt
```

To fine tune the Attention OCR model using a checkpoint:

```bash
python train.py \
    --checkpoint=model.ckpt-399731
```


## Evaluation

```bash
python eval.py \
    --train_log_dir=train_logs \
    --eval_log_dir=eval_logs \
    --num_batches=1
```


## Inference

```bash
python infer.py \
    --checkpoint=model.ckpt-399731 \
    --image_path_pattern=datasets/data/fsns/testdata/fsns_train_%02d.png
```
