# -*- coding: utf-8 -*-
"""
YOLOv8用データセットの準備
dataset/ok, dataset/ng から yolo_dataset/ にコピーして訓練/検証に分割
"""

import os
import shutil
import random
from pathlib import Path
import yaml
from config import DATASET_SETTINGS


def prepare_yolo_dataset():
    """
    YOLOv8用のデータセットを準備

    dataset/ok, dataset/ng から画像とラベルをコピーし、
    訓練用と検証用に分割
    """
    print("YOLOv8データセットを準備中...")

    # ディレクトリを作成
    os.makedirs(DATASET_SETTINGS['yolo_images_train'], exist_ok=True)
    os.makedirs(DATASET_SETTINGS['yolo_images_val'], exist_ok=True)
    os.makedirs(DATASET_SETTINGS['yolo_labels_train'], exist_ok=True)
    os.makedirs(DATASET_SETTINGS['yolo_labels_val'], exist_ok=True)

    # OK画像とNG画像を収集
    ok_images = list(Path(DATASET_SETTINGS['ok_dir']).glob("*.jpg"))
    ng_images = list(Path(DATASET_SETTINGS['ng_dir']).glob("*.jpg"))

    all_images = ok_images + ng_images

    print(f"  OK画像: {len(ok_images)}枚")
    print(f"  NG画像: {len(ng_images)}枚")
    print(f"  合計: {len(all_images)}枚")

    # シャッフル
    random.shuffle(all_images)

    # 訓練/検証に分割
    split_index = int(len(all_images) * DATASET_SETTINGS['train_val_split'])
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    print(f"  訓練データ: {len(train_images)}枚")
    print(f"  検証データ: {len(val_images)}枚")

    # ファイルをコピー
    def copy_files(image_list, images_dir, labels_dir):
        for img_path in image_list:
            # 画像をコピー
            dst_img = os.path.join(images_dir, img_path.name)
            shutil.copy(str(img_path), dst_img)

            # ラベルファイルをコピー（存在する場合）
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                dst_label = os.path.join(labels_dir, label_path.name)
                shutil.copy(str(label_path), dst_label)
            else:
                # ラベルがない場合は空ファイルを作成（OK品）
                dst_label = os.path.join(labels_dir, img_path.stem + '.txt')
                open(dst_label, 'w').close()

    print("  ファイルをコピー中...")
    copy_files(train_images, DATASET_SETTINGS['yolo_images_train'],
              DATASET_SETTINGS['yolo_labels_train'])
    copy_files(val_images, DATASET_SETTINGS['yolo_images_val'],
              DATASET_SETTINGS['yolo_labels_val'])

    # dataset.yamlを作成
    dataset_yaml = {
        'path': os.path.abspath(DATASET_SETTINGS['yolo_dataset_dir']),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # クラス数
        'names': ['wrinkle']  # クラス名
    }

    yaml_path = os.path.join(DATASET_SETTINGS['yolo_dataset_dir'], 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"  dataset.yamlを作成: {yaml_path}")
    print("データセット準備完了！")
    print(f"\n学習開始コマンド:")
    print(f"  yolo task=segment mode=train model=yolov8n-seg.pt data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    prepare_yolo_dataset()
