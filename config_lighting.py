# -*- coding: utf-8 -*-
"""
照明差分法システム用設定ファイル
"""

# カメラ設定（元のconfig.pyから継承）
CAMERA_SETTINGS = {
    # 通常モード設定
    'exposure_time': 10000,  # 10ms
    'gain': 12.0,
    'brightness': 128,

    # 画像サイズ
    'width': 1920,
    'height': 1200,

    # フレームレート
    'fps': 30,
}

# 照明差分法のパラメータ
LIGHTING_DETECTION_PARAMS = {
    # 差分閾値（シワと柄を区別する閾値）
    'difference_threshold': 20,  # 差分の閾値（0-255）

    # 横シワの最小長さ
    'min_wrinkle_length': 20,  # ピクセル

    # 照明切り替え後の安定待ち時間
    'stabilization_time': 0.1,  # 秒

    # 位置合わせを使用するか
    'use_alignment': True,

    # 高度な検出を使用するか（Sobel + 横シワ強調）
    'use_advanced_detection': True,

    # Sobelフィルタの閾値（上方照明画像での横線検出）
    'sobel_threshold': 10,

    # NG判定閾値
    'ng_wrinkle_ratio': 0.5,  # シワ面積率が0.5%以上でNG
    'ng_wrinkle_count': 3,    # シワが3本以上でNG
}

# GUI設定
GUI_SETTINGS_LIGHTING = {
    'window_title': 'シャンパンボトル シワ検査システム - 照明差分法',
    'preview_width': 640,
    'preview_height': 480,
}

# ファイル保存設定
SAVE_SETTINGS_LIGHTING = {
    'results_dir': 'results_lighting',
    'debug_dir': 'output/debug_lighting',
    'save_debug_images': True,
    'save_result_images': True,
    'log_file': 'results_lighting/detection_log.csv',
}

# 照明制御設定
LIGHTING_CONTROL_SETTINGS = {
    # GPIO制御を使用するか（Raspberry Pi等）
    'use_gpio': False,

    # GPIOピン番号（BCM番号）
    'coaxial_pin': 17,
    'top_pin': 27,
}
