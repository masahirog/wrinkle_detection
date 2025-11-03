# -*- coding: utf-8 -*-
"""
ユーティリティ関数
ログ記録、画像保存、時刻処理など
"""

import os
import csv
import json
from datetime import datetime
import cv2
from config import SAVE_SETTINGS

# CLAHE（適応的ヒストグラム平坦化）
_clahe = None


def get_clahe():
    """CLAHEオブジェクトを取得（シングルトン）"""
    global _clahe
    if _clahe is None:
        from config import DATASET_SETTINGS
        clip_limit = DATASET_SETTINGS.get('clahe_clip_limit', 2.0)
        tile_size = DATASET_SETTINGS.get('clahe_tile_size', 8)
        _clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return _clahe


def apply_clahe(image):
    """
    CLAHE（適応的ヒストグラム平坦化）を適用
    白いラベルと黒いラベルの両方でシワが見えるように補正

    Args:
        image: 入力画像（BGR）

    Returns:
        corrected: 補正後の画像（BGR）
    """
    clahe = get_clahe()

    # カラー画像の場合、L*a*b*色空間のLチャンネルに適用
    if len(image.shape) == 3:
        # BGR → Lab色空間に変換
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # L（明度）チャンネルにCLAHEを適用
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Lab → BGRに戻す
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # グレースケールの場合
        corrected = clahe.apply(image)

    return corrected


def ensure_directories():
    """必要なディレクトリを作成"""
    os.makedirs(SAVE_SETTINGS['test_images_dir'], exist_ok=True)
    os.makedirs(SAVE_SETTINGS['results_dir'], exist_ok=True)
    os.makedirs(SAVE_SETTINGS['debug_dir'], exist_ok=True)

    # データセットディレクトリも作成
    from config import DATASET_SETTINGS
    os.makedirs(DATASET_SETTINGS['ok_dir'], exist_ok=True)
    os.makedirs(DATASET_SETTINGS['ng_dir'], exist_ok=True)


def get_timestamp():
    """現在時刻を文字列で取得"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_image(image, result, count):
    """
    検査結果画像を保存

    Args:
        image: 保存する画像
        result: "OK" or "NG"
        count: 検出された線の数

    Returns:
        保存したファイルパス
    """
    # OK品を保存しない設定の場合はスキップ
    if result == "OK" and not SAVE_SETTINGS['save_ok_images']:
        return None

    # NG品を保存しない設定の場合はスキップ
    if result == "NG" and not SAVE_SETTINGS['save_ng_images']:
        return None

    timestamp = get_timestamp()
    filename = f"{result}_{timestamp}_lines{count}.jpg"
    filepath = os.path.join(SAVE_SETTINGS['results_dir'], filename)

    cv2.imwrite(filepath, image)
    return filepath


def init_log_file():
    """ログファイルを初期化（ヘッダー作成）"""
    ensure_directories()

    log_path = SAVE_SETTINGS['log_file']

    # ファイルが存在しない場合のみヘッダーを書き込む
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['日時', '判定', '検出線数', '画像パス'])


def log_result(result, count, image_path):
    """
    検査結果をログファイルに記録

    Args:
        result: "OK" or "NG"
        count: 検出された線の数
        image_path: 保存した画像のパス
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_path = SAVE_SETTINGS['log_file']

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, result, count, image_path if image_path else ''])


def save_debug_images(debug_images, result, count, camera_info=None, detection_params=None):
    """
    デバッグ画像と検査情報を保存

    Args:
        debug_images: デバッグ画像の辞書
        result: "OK" or "NG"
        count: 検出された線の数
        camera_info: カメラ設定情報の辞書（オプション）
        detection_params: 検出パラメータの辞書（オプション）

    Returns:
        保存したディレクトリパス
    """
    # デバッグ画像を保存しない設定の場合はスキップ
    if not SAVE_SETTINGS.get('save_debug_images', True):
        return None

    timestamp = get_timestamp()
    timestamp_readable = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    debug_dir = SAVE_SETTINGS['debug_dir']

    # タイムスタンプごとのサブディレクトリを作成
    session_dir = os.path.join(debug_dir, f"{timestamp}_{result}_lines{count}")
    os.makedirs(session_dir, exist_ok=True)

    # 各デバッグ画像を保存
    for key, image in debug_images.items():
        filename = f"{key}.jpg"
        filepath = os.path.join(session_dir, filename)
        cv2.imwrite(filepath, image)

    # 検査情報をJSONで保存
    inspection_info = {
        "timestamp": timestamp_readable,
        "result": result,
        "detected_lines": count,
        "camera_settings": camera_info if camera_info else {},
        "detection_params": detection_params if detection_params else {}
    }

    info_path = os.path.join(session_dir, "inspection_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(inspection_info, f, ensure_ascii=False, indent=2)

    return session_dir


def resize_for_display(image, target_width, target_height):
    """
    表示用に画像をリサイズ（アスペクト比維持）

    Args:
        image: 元画像
        target_width: 目標幅
        target_height: 目標高さ

    Returns:
        リサイズされた画像
    """
    h, w = image.shape[:2]

    # アスペクト比を計算
    aspect = w / h
    target_aspect = target_width / target_height

    if aspect > target_aspect:
        # 幅に合わせる
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        # 高さに合わせる
        new_h = target_height
        new_w = int(target_height * aspect)

    resized = cv2.resize(image, (new_w, new_h))
    return resized


def save_dataset_image(image, label):
    """
    学習データセット用の画像を保存

    Args:
        image: 保存する画像
        label: "ok" or "ng"

    Returns:
        保存したファイルパス
    """
    from config import DATASET_SETTINGS

    # 保存先ディレクトリ
    if label.lower() == "ok":
        save_dir = DATASET_SETTINGS['ok_dir']
    elif label.lower() == "ng":
        save_dir = DATASET_SETTINGS['ng_dir']
    else:
        raise ValueError(f"Invalid label: {label}")

    # 既存の画像数を取得
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
    next_number = len(existing_files) + 1

    # ファイル名を生成
    timestamp = get_timestamp()
    filename = f"{label}_{next_number:04d}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)

    # 保存
    cv2.imwrite(filepath, image)
    print(f"データセット画像保存: {filepath}")

    return filepath


def get_dataset_count():
    """
    現在のデータセット枚数を取得

    Returns:
        ok_count: OK品の枚数
        ng_count: NG品の枚数
    """
    from config import DATASET_SETTINGS

    ok_count = len([f for f in os.listdir(DATASET_SETTINGS['ok_dir']) if f.endswith(('.jpg', '.png'))])
    ng_count = len([f for f in os.listdir(DATASET_SETTINGS['ng_dir']) if f.endswith(('.jpg', '.png'))])

    return ok_count, ng_count
