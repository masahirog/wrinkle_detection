# -*- coding: utf-8 -*-
"""
設定ファイル
カメラパラメータと検出アルゴリズムの設定
"""

# カメラ設定
CAMERA_SETTINGS = {
    # 通常モード設定
    'exposure_time': 10000,  # 10ms（初期値、調整可能範囲: 5000-20000）
    'gain': 12.0,  # 初期値、調整可能範囲: 0-20
    'brightness': 128,  # 初期値、調整可能範囲: 0-255

    # ブレ防止モード設定（動くボトル用）
    'exposure_time_fast': 2000,   # 2ms（高速シャッター）
    'gain_fast': 18.0,            # ゲイン上げて明るさ確保
    'brightness_fast': 180,       # 明るさ上げる

    # 画像サイズ
    'width': 1920,
    'height': 1200,

    # フレームレート
    'fps': 30,
}

# シワ検出パラメータ
DETECTION_PARAMS = {
    # グレースケール変換
    'use_grayscale': True,

    # 2値化閾値
    'binary_threshold': 200,  # 明るい部分を除去（露出オーバーで影を強調）

    # モルフォロジー処理
    'morph_kernel_size': (3, 3),

    # Cannyエッジ検出
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,

    # Hough線検出
    'hough_threshold': 50,
    'hough_min_line_length': 20,
    'hough_max_line_gap': 5,

    # シワ判定閾値（検出された線の本数）
    'wrinkle_line_threshold': 5,  # 5本以上の線が検出されたらNG

    # 輪郭検出パラメータ
    'use_contour_detection': True,  # 輪郭検出を使用するか
    'contour_deviation_threshold': 10.0,  # 輪郭の変動閾値（ピクセル）
    'contour_smoothing_window': 50,  # 輪郭の平滑化ウィンドウサイズ

    # ボトル形状抽出パラメータ
    'canny_low_threshold_bottle': 30,  # Canny下限閾値（ボトル抽出用）
    'canny_high_threshold_bottle': 150,  # Canny上限閾値（ボトル抽出用）
    'use_convex_hull': False,  # 凸包を使用（輪郭を滑らかにする、シワも消える）
    'morph_kernel_size_bottle': 5,  # モルフォロジー処理のカーネルサイズ
}

# ファイル保存設定
SAVE_SETTINGS = {
    'test_images_dir': 'test_images',
    'results_dir': 'results',
    'debug_dir': 'output/debug',  # デバッグ画像保存先
    'save_ok_images': False,  # OK品の画像を保存するか
    'save_ng_images': True,   # NG品の画像を保存するか
    'save_debug_images': True,  # デバッグ画像を保存するか
    'log_file': 'results/detection_log.csv',
}

# 学習データセット設定（YOLOv8用）
DATASET_SETTINGS = {
    'dataset_dir': 'dataset',  # データセットのルートディレクトリ
    'ok_dir': 'dataset/ok',    # OK品の画像保存先
    'ng_dir': 'dataset/ng',    # NG品の画像保存先
    'target_ok_count': 1500,   # 目標OK品枚数（YOLOv8学習用）
    'target_ng_count': 1500,   # 目標NG品枚数（YOLOv8学習用）

    # YOLOv8用のディレクトリ構成
    'yolo_dataset_dir': 'yolo_dataset',
    'yolo_images_train': 'yolo_dataset/images/train',
    'yolo_images_val': 'yolo_dataset/images/val',
    'yolo_labels_train': 'yolo_dataset/labels/train',
    'yolo_labels_val': 'yolo_dataset/labels/val',
    'train_val_split': 0.8,  # 80%を訓練、20%を検証

    # CLAHE（適応的ヒストグラム平坦化）設定
    'use_clahe': True,  # CLAHEを使用するか
    'clahe_clip_limit': 1.0,  # クリップ限界（小さいほど白飛び抑制）
    'clahe_tile_size': 4,  # タイルサイズ（小さいほど細かく調整）
}

# GUI設定
GUI_SETTINGS = {
    'window_title': 'シャンパンボトル シワ検査システム - Phase 1',
    'preview_width': 640,
    'preview_height': 480,
}
