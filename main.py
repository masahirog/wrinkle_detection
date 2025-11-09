# -*- coding: utf-8 -*-
"""
メインプログラム
シャンパンボトル シワ検査システム Phase 1
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import signal
import sys
import json
import os
import numpy as np

from camera_control import StCameraControl
from image_processing import detect_wrinkles, draw_result_on_image, detect_bottle_with_yolo, extract_bottle_mask, create_filled_bottle_image
from utils import ensure_directories, init_log_file, save_image, log_result, resize_for_display, save_debug_images, save_dataset_image, get_dataset_count
from config import CAMERA_SETTINGS, DETECTION_PARAMS, GUI_SETTINGS, SAVE_SETTINGS, DATASET_SETTINGS


class WrinkleDetectionApp:
    """シワ検査アプリケーション"""

    def __init__(self, root):
        """初期化"""
        self.root = root
        self.root.title(GUI_SETTINGS['window_title'])

        # カメラコントロール
        self.camera = StCameraControl()

        # 状態フラグ
        self.is_running = False
        self.is_inspecting = False
        self.auto_capture_running = False  # 自動撮影中かどうか
        self.last_capture_time = 0  # 最後に撮影した時刻
        self.bottle_detected = False  # 現在ボトルが検出されているか

        # 現在のフレーム
        self.current_frame_original = None  # オリジナル画像
        self.current_frame_corrected = None  # プリセット適用後（保存用）

        # CLAHEパラメータ
        self.clahe_clip_limit = tk.DoubleVar(value=DATASET_SETTINGS['clahe_clip_limit'])
        self.clahe_tile_size = tk.IntVar(value=DATASET_SETTINGS['clahe_tile_size'])

        # カメラパラメータ
        self.param_exposure = tk.IntVar(value=CAMERA_SETTINGS['exposure_time'])
        self.param_gain = tk.DoubleVar(value=CAMERA_SETTINGS['gain'])
        self.param_digital_gain = tk.DoubleVar(value=1.0)
        self.param_black_level = tk.IntVar(value=0)
        self.param_wb_red = tk.DoubleVar(value=1.0)
        self.param_wb_blue = tk.DoubleVar(value=1.0)
        self.param_highlight_comp = tk.DoubleVar(value=1.0)  # 1.0=圧縮なし
        self.param_saturation = tk.DoubleVar(value=1.0)  # 1.0=フルカラー
        self.adaptive_processing = tk.BooleanVar(value=False)  # 適応的処理OFF
        self.rotation_angle = tk.IntVar(value=0)  # 回転角度
        self.bayer_pattern = tk.StringVar(value="BG")  # 正常動作するパターン

        # 統計情報
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0

        # 利用可能なカメラリスト
        self.available_cameras = []

        # カメラの基準パラメータ（起動時の値を記録）
        self.base_exposure = CAMERA_SETTINGS['exposure_time']
        self.base_gain = CAMERA_SETTINGS['gain']

        # プリセット管理
        self.preset_file = "presets.json"
        self.presets = self.load_presets()

        # 自動プリセット切り替え（4段階）
        self.auto_preset_enabled = tk.BooleanVar(value=False)  # 自動切り替えON/OFF
        self.auto_preset_config_file = "auto_preset_config.json"

        # 4段階の閾値とプリセット
        self.threshold1 = tk.IntVar(value=60)   # 範囲1と2の境界
        self.threshold2 = tk.IntVar(value=100)  # 範囲2と3の境界
        self.threshold3 = tk.IntVar(value=140)  # 範囲3と4の境界
        self.preset_range1 = tk.IntVar(value=0)  # 最暗（0～閾値1）
        self.preset_range2 = tk.IntVar(value=1)  # 暗（閾値1～閾値2）
        self.preset_range3 = tk.IntVar(value=2)  # 明（閾値2～閾値3）
        self.preset_range4 = tk.IntVar(value=3)  # 最明（閾値3～255）

        self.luminance_hysteresis = tk.IntVar(value=10)  # ヒステリシス幅（パタパタ防止）
        self.current_auto_preset = None  # 現在適用中のプリセット番号
        self.current_range = None  # 現在の輝度範囲（1-4）

        # 設定を読み込み
        self.load_auto_preset_config()

        # ディレクトリとログファイルの初期化
        ensure_directories()
        init_log_file()

        # GUI構築
        self.build_gui()

        # ウィンドウ閉じるボタンのハンドラを設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # カメラを自動スキャン
        self.scan_cameras()

    def build_gui(self):
        """GUIを構築"""

        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左側: カメラプレビュー（2画面）
        preview_container = ttk.Frame(main_frame)
        preview_container.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky=(tk.N, tk.S))

        # 左：オリジナル画像（YOLO検出用）
        original_preview_frame = ttk.LabelFrame(preview_container, text="オリジナル画像（YOLO検出）", padding="10")
        original_preview_frame.grid(row=0, column=0, padx=5, pady=5)
        self.original_preview_label = ttk.Label(original_preview_frame)
        self.original_preview_label.pack()

        # 右：プリセット適用後の画像
        preset_preview_frame = ttk.LabelFrame(preview_container, text="プリセット適用後", padding="10")
        preset_preview_frame.grid(row=0, column=1, padx=5, pady=5)
        self.preset_preview_label = ttk.Label(preset_preview_frame)
        self.preset_preview_label.pack()

        # 右側: スクロール可能なコントロールエリア
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))

        # スクロールバー付きキャンバス
        canvas = tk.Canvas(right_frame, width=500, height=800)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # マウスホイールでスクロール
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # カメラ制御フレーム（scrollable_frame内に配置）
        camera_control_frame = ttk.LabelFrame(scrollable_frame, text="カメラ制御", padding="10")
        camera_control_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # カメラ選択
        ttk.Label(camera_control_frame, text="カメラ:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            camera_control_frame,
            textvariable=self.camera_var,
            state='readonly',
            width=30
        )
        self.camera_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_change)

        # カメラ再検出ボタン
        self.rescan_button = ttk.Button(camera_control_frame, text="再検出", command=self.scan_cameras)
        self.rescan_button.grid(row=0, column=3, padx=5, pady=5)

        # 画像回転選択
        rotation_frame = ttk.LabelFrame(camera_control_frame, text="画像回転", padding="5")
        rotation_frame.grid(row=1, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        ttk.Radiobutton(rotation_frame, text="0度", variable=self.rotation_angle,
                       value=0, command=self.on_rotation_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(rotation_frame, text="90度", variable=self.rotation_angle,
                       value=90, command=self.on_rotation_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(rotation_frame, text="180度", variable=self.rotation_angle,
                       value=180, command=self.on_rotation_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(rotation_frame, text="270度", variable=self.rotation_angle,
                       value=270, command=self.on_rotation_change).pack(side=tk.LEFT, padx=5)

        # モード切り替えボタン
        mode_button_frame = ttk.LabelFrame(camera_control_frame, text="撮影モード", padding="5")
        mode_button_frame.grid(row=2, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(mode_button_frame, text="通常モード", command=self.set_normal_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(mode_button_frame, text="ブレ防止モード", command=self.set_fast_mode).pack(side=tk.LEFT, padx=5)

        # プリセットボタン（5つ）
        preset_button_frame = ttk.LabelFrame(camera_control_frame, text="プリセット", padding="5")
        preset_button_frame.grid(row=3, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        # プリセット名とボタンを格納
        self.preset_labels = []

        for i in range(5):
            # プリセット名（クリックで編集可能）
            preset_name = self.presets[i].get('name', f'プリセット{i+1}')
            label = ttk.Label(preset_button_frame, text=preset_name, width=12,
                            relief=tk.RIDGE, cursor="hand2", anchor=tk.CENTER)
            label.grid(row=0, column=i*3, padx=2, pady=2)
            label.bind('<Double-Button-1>', lambda e, idx=i: self.rename_preset(idx))
            self.preset_labels.append(label)

            # 読み込みボタン
            load_btn = ttk.Button(preset_button_frame, text="読込", width=5,
                                 command=lambda idx=i: self.load_preset(idx))
            load_btn.grid(row=1, column=i*3, padx=2, pady=2)

            # 保存ボタン
            save_btn = ttk.Button(preset_button_frame, text="保存", width=5,
                                 command=lambda idx=i: self.save_preset(idx))
            save_btn.grid(row=2, column=i*3, padx=2, pady=2)

        # 自動プリセット切り替え（4段階）
        auto_preset_frame = ttk.LabelFrame(camera_control_frame, text="自動プリセット切り替え（4段階）", padding="5")
        auto_preset_frame.grid(row=4, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        # 自動切り替えON/OFF
        self.auto_preset_check = ttk.Checkbutton(auto_preset_frame, text="自動切り替えを有効化",
                                                 variable=self.auto_preset_enabled)
        self.auto_preset_check.grid(row=0, column=0, columnspan=4, padx=5, pady=2, sticky=tk.W)

        # 4段階の設定
        preset_names = [self.presets[i].get('name', f'プリセット{i+1}') for i in range(5)]

        # 範囲1（0～閾値1）
        ttk.Label(auto_preset_frame, text="範囲1（最暗）:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.preset_range1_combo = ttk.Combobox(auto_preset_frame, textvariable=self.preset_range1,
                                               values=list(range(5)), state='readonly', width=5)
        self.preset_range1_combo.grid(row=1, column=1, padx=5, pady=2)
        self.preset_range1_label = ttk.Label(auto_preset_frame, text=preset_names[0], width=12)
        self.preset_range1_label.grid(row=1, column=2, padx=5, pady=2)

        # 閾値1スライダー
        ttk.Label(auto_preset_frame, text="閾値1:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.threshold1_scale = ttk.Scale(auto_preset_frame, from_=0, to=255,
                                         variable=self.threshold1, orient=tk.HORIZONTAL, length=100)
        self.threshold1_scale.grid(row=2, column=1, padx=5, pady=2)
        self.threshold1_label = ttk.Label(auto_preset_frame, text=f"{self.threshold1.get()}")
        self.threshold1_label.grid(row=2, column=2, padx=5)
        self.threshold1.trace_add('write', lambda *args: self.threshold1_label.config(text=f"{self.threshold1.get()}"))

        # 範囲2（閾値1～閾値2）
        ttk.Label(auto_preset_frame, text="範囲2（暗）:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.preset_range2_combo = ttk.Combobox(auto_preset_frame, textvariable=self.preset_range2,
                                               values=list(range(5)), state='readonly', width=5)
        self.preset_range2_combo.grid(row=3, column=1, padx=5, pady=2)
        self.preset_range2_label = ttk.Label(auto_preset_frame, text=preset_names[1], width=12)
        self.preset_range2_label.grid(row=3, column=2, padx=5, pady=2)

        # 閾値2スライダー
        ttk.Label(auto_preset_frame, text="閾値2:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.threshold2_scale = ttk.Scale(auto_preset_frame, from_=0, to=255,
                                         variable=self.threshold2, orient=tk.HORIZONTAL, length=100)
        self.threshold2_scale.grid(row=4, column=1, padx=5, pady=2)
        self.threshold2_label = ttk.Label(auto_preset_frame, text=f"{self.threshold2.get()}")
        self.threshold2_label.grid(row=4, column=2, padx=5)
        self.threshold2.trace_add('write', lambda *args: self.threshold2_label.config(text=f"{self.threshold2.get()}"))

        # 範囲3（閾値2～閾値3）
        ttk.Label(auto_preset_frame, text="範囲3（明）:").grid(row=5, column=0, sticky=tk.W, padx=5)
        self.preset_range3_combo = ttk.Combobox(auto_preset_frame, textvariable=self.preset_range3,
                                               values=list(range(5)), state='readonly', width=5)
        self.preset_range3_combo.grid(row=5, column=1, padx=5, pady=2)
        self.preset_range3_label = ttk.Label(auto_preset_frame, text=preset_names[2], width=12)
        self.preset_range3_label.grid(row=5, column=2, padx=5, pady=2)

        # 閾値3スライダー
        ttk.Label(auto_preset_frame, text="閾値3:").grid(row=6, column=0, sticky=tk.W, padx=5)
        self.threshold3_scale = ttk.Scale(auto_preset_frame, from_=0, to=255,
                                         variable=self.threshold3, orient=tk.HORIZONTAL, length=100)
        self.threshold3_scale.grid(row=6, column=1, padx=5, pady=2)
        self.threshold3_label = ttk.Label(auto_preset_frame, text=f"{self.threshold3.get()}")
        self.threshold3_label.grid(row=6, column=2, padx=5)
        self.threshold3.trace_add('write', lambda *args: self.threshold3_label.config(text=f"{self.threshold3.get()}"))

        # 範囲4（閾値3～255）
        ttk.Label(auto_preset_frame, text="範囲4（最明）:").grid(row=7, column=0, sticky=tk.W, padx=5)
        self.preset_range4_combo = ttk.Combobox(auto_preset_frame, textvariable=self.preset_range4,
                                               values=list(range(5)), state='readonly', width=5)
        self.preset_range4_combo.grid(row=7, column=1, padx=5, pady=2)
        self.preset_range4_label = ttk.Label(auto_preset_frame, text=preset_names[3], width=12)
        self.preset_range4_label.grid(row=7, column=2, padx=5, pady=2)

        # ヒステリシス幅スライダー（パタパタ防止）
        ttk.Label(auto_preset_frame, text="ヒステリシス:").grid(row=8, column=0, sticky=tk.W, padx=5)
        self.luminance_hysteresis_scale = ttk.Scale(auto_preset_frame, from_=0, to=50,
                                                    variable=self.luminance_hysteresis,
                                                    orient=tk.HORIZONTAL, length=100)
        self.luminance_hysteresis_scale.grid(row=8, column=1, padx=5, pady=2)
        self.luminance_hysteresis_label = ttk.Label(auto_preset_frame, text=f"{self.luminance_hysteresis.get()}")
        self.luminance_hysteresis_label.grid(row=8, column=2, padx=5)
        self.luminance_hysteresis.trace_add('write', lambda *args: self.luminance_hysteresis_label.config(text=f"{self.luminance_hysteresis.get()}"))

        # 設定保存ボタン
        ttk.Button(auto_preset_frame, text="設定を保存", command=self.save_auto_preset_config).grid(row=9, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 現在選択中のプリセット表示
        ttk.Label(auto_preset_frame, text="現在選択中:").grid(row=10, column=0, sticky=tk.W, padx=5)
        self.current_preset_display = ttk.Label(auto_preset_frame, text="なし",
                                               font=('Arial', 10, 'bold'), foreground="blue")
        self.current_preset_display.grid(row=10, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)

        # プリセット選択変更時にラベルを更新
        self.preset_range1.trace_add('write', self.update_auto_preset_labels)
        self.preset_range2.trace_add('write', self.update_auto_preset_labels)
        self.preset_range3.trace_add('write', self.update_auto_preset_labels)
        self.preset_range4.trace_add('write', self.update_auto_preset_labels)

        # カメラ起動/停止ボタン
        self.start_button = ttk.Button(camera_control_frame, text="カメラ起動", command=self.start_camera)
        self.start_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.stop_button = ttk.Button(camera_control_frame, text="カメラ停止", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.grid(row=5, column=2, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # CLAHE調整スライダー
        clahe_frame = ttk.LabelFrame(camera_control_frame, text="CLAHE調整（白飛び・黒つぶれ対策）", padding="5")
        clahe_frame.grid(row=6, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        # CLAHEリセットボタン
        ttk.Button(clahe_frame, text="デフォルトに戻す",
                  command=self.reset_clahe_params).grid(row=0, column=3, padx=5, pady=2, sticky=tk.E)

        ttk.Label(clahe_frame, text="クリップ限界:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.clahe_clip_scale = ttk.Scale(clahe_frame, from_=0.5, to=4.0,
                                         variable=self.clahe_clip_limit,
                                         orient=tk.HORIZONTAL, length=150)
        self.clahe_clip_scale.grid(row=1, column=1, padx=5, pady=2)
        self.clahe_clip_label = ttk.Label(clahe_frame, text=f"{self.clahe_clip_limit.get():.1f}")
        self.clahe_clip_label.grid(row=1, column=2, padx=5)
        self.clahe_clip_limit.trace_add('write', lambda *args: self.clahe_clip_label.config(text=f"{self.clahe_clip_limit.get():.1f}"))

        ttk.Label(clahe_frame, text="タイルサイズ:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.clahe_tile_scale = ttk.Scale(clahe_frame, from_=2, to=16,
                                         variable=self.clahe_tile_size,
                                         orient=tk.HORIZONTAL, length=150)
        self.clahe_tile_scale.grid(row=2, column=1, padx=5, pady=2)
        self.clahe_tile_label = ttk.Label(clahe_frame, text=f"{self.clahe_tile_size.get()}")
        self.clahe_tile_label.grid(row=2, column=2, padx=5)
        self.clahe_tile_size.trace_add('write', lambda *args: self.clahe_tile_label.config(text=f"{self.clahe_tile_size.get()}"))

        # カメラパラメータ調整スライダー
        camera_param_frame = ttk.LabelFrame(camera_control_frame, text="カメラパラメータ調整", padding="5")
        camera_param_frame.grid(row=7, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        # リセットボタン
        ttk.Button(camera_param_frame, text="デフォルトに戻す",
                  command=self.reset_camera_params).grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky=tk.E)

        # 露出時間
        ttk.Label(camera_param_frame, text="露出時間:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.exposure_scale = ttk.Scale(camera_param_frame, from_=2000, to=20000,
                                       variable=self.param_exposure,
                                       orient=tk.HORIZONTAL, length=150,
                                       command=self.on_camera_param_change)
        self.exposure_scale.grid(row=1, column=1, padx=5, pady=2)
        self.exposure_label = ttk.Label(camera_param_frame, text=f"{self.param_exposure.get()}μs")
        self.exposure_label.grid(row=1, column=2, padx=5)

        # ゲイン（アナログ）
        ttk.Label(camera_param_frame, text="ゲイン:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.gain_scale = ttk.Scale(camera_param_frame, from_=0, to=24,
                                   variable=self.param_gain,
                                   orient=tk.HORIZONTAL, length=150,
                                   command=self.on_camera_param_change)
        self.gain_scale.grid(row=2, column=1, padx=5, pady=2)
        self.gain_label = ttk.Label(camera_param_frame, text=f"{self.param_gain.get():.1f}dB")
        self.gain_label.grid(row=2, column=2, padx=5)

        # デジタルゲイン
        ttk.Label(camera_param_frame, text="デジタルゲイン:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.digital_gain_scale = ttk.Scale(camera_param_frame, from_=1.0, to=2.0,
                                           variable=self.param_digital_gain,
                                           orient=tk.HORIZONTAL, length=150,
                                           command=self.on_camera_param_change)
        self.digital_gain_scale.grid(row=3, column=1, padx=5, pady=2)
        self.digital_gain_label = ttk.Label(camera_param_frame, text=f"{self.param_digital_gain.get():.2f}x")
        self.digital_gain_label.grid(row=3, column=2, padx=5)

        # 黒レベル
        ttk.Label(camera_param_frame, text="黒レベル:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.black_level_scale = ttk.Scale(camera_param_frame, from_=0, to=100,
                                          variable=self.param_black_level,
                                          orient=tk.HORIZONTAL, length=150,
                                          command=self.on_camera_param_change)
        self.black_level_scale.grid(row=4, column=1, padx=5, pady=2)
        self.black_level_label = ttk.Label(camera_param_frame, text=f"{self.param_black_level.get()}")
        self.black_level_label.grid(row=4, column=2, padx=5)

        # ホワイトバランス（赤）- ソフトウェア処理
        ttk.Label(camera_param_frame, text="WB 赤(SW):").grid(row=5, column=0, sticky=tk.W, padx=5)
        self.wb_red_scale = ttk.Scale(camera_param_frame, from_=0.5, to=4.0,
                                     variable=self.param_wb_red,
                                     orient=tk.HORIZONTAL, length=150,
                                     command=self.on_camera_param_change)
        self.wb_red_scale.grid(row=5, column=1, padx=5, pady=2)
        self.wb_red_label = ttk.Label(camera_param_frame, text=f"{self.param_wb_red.get():.2f}")
        self.wb_red_label.grid(row=5, column=2, padx=5)

        # ホワイトバランス（青）- ソフトウェア処理
        ttk.Label(camera_param_frame, text="WB 青(SW):").grid(row=6, column=0, sticky=tk.W, padx=5)
        self.wb_blue_scale = ttk.Scale(camera_param_frame, from_=0.5, to=4.0,
                                      variable=self.param_wb_blue,
                                      orient=tk.HORIZONTAL, length=150,
                                      command=self.on_camera_param_change)
        self.wb_blue_scale.grid(row=6, column=1, padx=5, pady=2)
        self.wb_blue_label = ttk.Label(camera_param_frame, text=f"{self.param_wb_blue.get():.2f}")
        self.wb_blue_label.grid(row=6, column=2, padx=5)

        # ハイライト圧縮（白飛び抑制）
        ttk.Label(camera_param_frame, text="ハイライト圧縮:").grid(row=7, column=0, sticky=tk.W, padx=5)
        self.highlight_comp_scale = ttk.Scale(camera_param_frame, from_=0.0, to=1.0,
                                             variable=self.param_highlight_comp,
                                             orient=tk.HORIZONTAL, length=150,
                                             command=self.on_camera_param_change)
        self.highlight_comp_scale.grid(row=7, column=1, padx=5, pady=2)
        self.highlight_comp_label = ttk.Label(camera_param_frame, text=f"{self.param_highlight_comp.get():.2f}")
        self.highlight_comp_label.grid(row=7, column=2, padx=5)

        # 彩度調整
        ttk.Label(camera_param_frame, text="彩度:").grid(row=8, column=0, sticky=tk.W, padx=5)
        self.saturation_scale = ttk.Scale(camera_param_frame, from_=0.0, to=1.0,
                                         variable=self.param_saturation,
                                         orient=tk.HORIZONTAL, length=150,
                                         command=self.on_camera_param_change)
        self.saturation_scale.grid(row=8, column=1, padx=5, pady=2)
        self.saturation_label = ttk.Label(camera_param_frame, text=f"{self.param_saturation.get():.2f}")
        self.saturation_label.grid(row=8, column=2, padx=5)

        # 適応的処理
        adaptive_check = ttk.Checkbutton(camera_param_frame, text="適応的処理（明暗領域を自動調整）",
                                        variable=self.adaptive_processing)
        adaptive_check.grid(row=9, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        # データ収集フレーム（scrollable_frame内に配置）
        control_frame = ttk.LabelFrame(scrollable_frame, text="データ収集", padding="10")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))

        # 手動撮影ボタン
        manual_frame = ttk.LabelFrame(control_frame, text="手動撮影", padding="10")
        manual_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.collect_ok_button = ttk.Button(manual_frame, text="OK品として保存", command=lambda: self.save_to_dataset("ok"), state=tk.DISABLED)
        self.collect_ok_button.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.collect_ng_button = ttk.Button(manual_frame, text="NG品として保存", command=lambda: self.save_to_dataset("ng"), state=tk.DISABLED)
        self.collect_ng_button.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 自動撮影設定
        auto_frame = ttk.LabelFrame(control_frame, text="自動撮影", padding="10")
        auto_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # 撮影間隔スライダー
        ttk.Label(auto_frame, text="撮影間隔 (秒):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.capture_interval_var = tk.DoubleVar(value=1.0)
        self.capture_interval_scale = ttk.Scale(
            auto_frame,
            from_=0.5,
            to=5.0,
            variable=self.capture_interval_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=self.on_interval_change
        )
        self.capture_interval_scale.grid(row=0, column=1, padx=5, pady=5)
        self.capture_interval_label = ttk.Label(auto_frame, text="1.0")
        self.capture_interval_label.grid(row=0, column=2, pady=5)

        # 自動撮影開始/停止ボタン
        self.auto_start_button = ttk.Button(auto_frame, text="自動撮影 開始", command=self.start_auto_capture, state=tk.DISABLED)
        self.auto_start_button.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.auto_stop_button = ttk.Button(auto_frame, text="自動撮影 停止", command=self.stop_auto_capture, state=tk.DISABLED)
        self.auto_stop_button.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 自動撮影ステータス
        self.auto_status_label = ttk.Label(auto_frame, text="待機中", font=('Arial', 10))
        self.auto_status_label.grid(row=2, column=0, columnspan=3, pady=5)

        # 撮影インジケーター（フラッシュ効果）
        self.capture_indicator = ttk.Label(auto_frame, text="", font=('Arial', 12, 'bold'),
                                          background='white', width=20)
        self.capture_indicator.grid(row=3, column=0, columnspan=3, pady=5)

        # データ収集状況とレビュー
        data_frame = ttk.LabelFrame(control_frame, text="データ管理", padding="10")
        data_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.dataset_label = ttk.Label(data_frame, text="OK: 0/50 | NG: 0/50", font=('Arial', 10))
        self.dataset_label.grid(row=0, column=0, columnspan=2, pady=5)

        self.review_button = ttk.Button(data_frame, text="OK画像をレビュー（NG品を振り分け）", command=self.open_review_window)
        self.review_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # データ収集状況を更新
        self.update_dataset_count()


    def start_camera(self):
        """カメラを起動"""
        # 選択されたカメラのインデックスを取得
        selected_index = self.camera_combo.current()
        if selected_index < 0 or not self.available_cameras:
            messagebox.showerror("エラー", "カメラを選択してください")
            return

        camera_info = self.available_cameras[selected_index]
        camera_index = camera_info['index']

        # カメラを開く
        if self.camera.open(camera_index):
            self.is_running = True

            # カメラ設定を適用（スライダーの値）
            self.camera.set_bayer_pattern(self.bayer_pattern.get())
            self.camera.set_exposure(self.param_exposure.get())
            self.camera.set_gain(self.param_gain.get())
            self.camera.set_digital_gain(self.param_digital_gain.get())
            self.camera.set_black_level(self.param_black_level.get())
            self.camera.set_white_balance(self.param_wb_red.get(), self.param_wb_blue.get())  # ソフトウェア処理
            self.camera.set_highlight_compression(self.param_highlight_comp.get())
            self.camera.set_saturation(self.param_saturation.get())
            self.camera.set_rotation(self.rotation_angle.get())

            # ボタン状態変更
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.collect_ok_button.config(state=tk.NORMAL)
            self.collect_ng_button.config(state=tk.NORMAL)
            self.auto_start_button.config(state=tk.NORMAL)
            self.camera_combo.config(state='readonly')  # カメラ選択は有効のまま

            # プレビュー開始
            self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
            self.preview_thread.start()

            # messagebox.showinfo("成功", f"カメラ {camera_index} を起動しました")
        else:
            messagebox.showerror("エラー", "カメラの起動に失敗しました")

    def stop_camera(self):
        """カメラを停止"""
        self.is_running = False
        time.sleep(0.5)  # プレビューループの終了を待つ

        self.camera.close()

        # 自動撮影が動いていたら停止
        if self.auto_capture_running:
            self.stop_auto_capture()

        # ボタン状態変更
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.collect_ok_button.config(state=tk.DISABLED)
        self.collect_ng_button.config(state=tk.DISABLED)
        self.auto_start_button.config(state=tk.DISABLED)
        self.auto_stop_button.config(state=tk.DISABLED)
        self.camera_combo.config(state='readonly')

        # messagebox.showinfo("成功", "カメラを停止しました")

    def preview_loop(self):
        """プレビューループ（別スレッド）- 2画面表示"""
        while self.is_running:
            frame = self.camera.capture_frame()

            if frame is not None:
                # オリジナル画像を保存
                self.current_frame_original = frame.copy()

                # YOLOでボトル検出（オリジナル画像で高精度検出）
                yolo_boxes = []
                try:
                    yolo_boxes, _ = detect_bottle_with_yolo(frame)
                except:
                    pass

                # 自動プリセット切り替え（UIの値のみ変更、カメラには適用しない）
                if self.auto_preset_enabled.get() and yolo_boxes and len(yolo_boxes) > 0:
                    # ボトル領域の平均輝度を計算
                    x1, y1, x2, y2 = yolo_boxes[0]  # 最初のボトルを使用
                    bottle_region = frame[y1:y2, x1:x2]

                    # LAB色空間でL（輝度）チャンネルの平均値を計算
                    lab = cv2.cvtColor(bottle_region, cv2.COLOR_BGR2LAB)
                    avg_luminance = np.mean(lab[:, :, 0])

                    # 4段階切り替え用の閾値とヒステリシスを取得
                    threshold1 = self.threshold1.get()
                    threshold2 = self.threshold2.get()
                    threshold3 = self.threshold3.get()
                    hysteresis = self.luminance_hysteresis.get()

                    # デバッグ: 輝度値を常に表示
                    print(f"[DEBUG] 輝度: {avg_luminance:.1f}, 閾値1: {threshold1}, 閾値2: {threshold2}, 閾値3: {threshold3}, ヒステリシス: {hysteresis}, 現在の状態: {self.current_range}")

                    # 新しい範囲を判定
                    new_range = None
                    if self.current_range is None:
                        # 初回は単純な閾値比較
                        if avg_luminance < threshold1:
                            new_range = "range1"
                            print(f"[DEBUG] 初回判定 → range1 (輝度 {avg_luminance:.1f} < 閾値1 {threshold1})")
                        elif avg_luminance < threshold2:
                            new_range = "range2"
                            print(f"[DEBUG] 初回判定 → range2 (輝度 {avg_luminance:.1f} < 閾値2 {threshold2})")
                        elif avg_luminance < threshold3:
                            new_range = "range3"
                            print(f"[DEBUG] 初回判定 → range3 (輝度 {avg_luminance:.1f} < 閾値3 {threshold3})")
                        else:
                            new_range = "range4"
                            print(f"[DEBUG] 初回判定 → range4 (輝度 {avg_luminance:.1f} >= 閾値3 {threshold3})")
                    else:
                        # ヒステリシスを使った判定（現在の範囲に応じて切り替え閾値を調整）
                        if self.current_range == "range1":
                            # 範囲1 → 範囲2への切り替えのみ可能
                            if avg_luminance >= threshold1 + hysteresis:
                                new_range = "range2"
                                print(f"[DEBUG] range1 → range2 切り替え (輝度 {avg_luminance:.1f} >= {threshold1 + hysteresis})")
                            else:
                                new_range = "range1"
                                print(f"[DEBUG] range1 維持 (輝度 {avg_luminance:.1f} < {threshold1 + hysteresis})")
                        elif self.current_range == "range2":
                            # 範囲2 → 範囲1 または 範囲3への切り替えが可能
                            if avg_luminance < threshold1 - hysteresis:
                                new_range = "range1"
                                print(f"[DEBUG] range2 → range1 切り替え (輝度 {avg_luminance:.1f} < {threshold1 - hysteresis})")
                            elif avg_luminance >= threshold2 + hysteresis:
                                new_range = "range3"
                                print(f"[DEBUG] range2 → range3 切り替え (輝度 {avg_luminance:.1f} >= {threshold2 + hysteresis})")
                            else:
                                new_range = "range2"
                                print(f"[DEBUG] range2 維持 (輝度 {avg_luminance:.1f})")
                        elif self.current_range == "range3":
                            # 範囲3 → 範囲2 または 範囲4への切り替えが可能
                            if avg_luminance < threshold2 - hysteresis:
                                new_range = "range2"
                                print(f"[DEBUG] range3 → range2 切り替え (輝度 {avg_luminance:.1f} < {threshold2 - hysteresis})")
                            elif avg_luminance >= threshold3 + hysteresis:
                                new_range = "range4"
                                print(f"[DEBUG] range3 → range4 切り替え (輝度 {avg_luminance:.1f} >= {threshold3 + hysteresis})")
                            else:
                                new_range = "range3"
                                print(f"[DEBUG] range3 維持 (輝度 {avg_luminance:.1f})")
                        elif self.current_range == "range4":
                            # 範囲4 → 範囲3への切り替えのみ可能
                            if avg_luminance < threshold3 - hysteresis:
                                new_range = "range3"
                                print(f"[DEBUG] range4 → range3 切り替え (輝度 {avg_luminance:.1f} < {threshold3 - hysteresis})")
                            else:
                                new_range = "range4"
                                print(f"[DEBUG] range4 維持 (輝度 {avg_luminance:.1f} >= {threshold3 - hysteresis})")

                    # プリセットを選択
                    if new_range == "range1":
                        selected_preset_idx = self.preset_range1.get()
                        range_display = "範囲1（最暗）"
                    elif new_range == "range2":
                        selected_preset_idx = self.preset_range2.get()
                        range_display = "範囲2（暗）"
                    elif new_range == "range3":
                        selected_preset_idx = self.preset_range3.get()
                        range_display = "範囲3（明）"
                    else:  # range4
                        selected_preset_idx = self.preset_range4.get()
                        range_display = "範囲4（最明）"

                    preset_name = self.presets[selected_preset_idx].get('name', f'プリセット{selected_preset_idx+1}')

                    # 現在のプリセット表示を常に更新（輝度をリアルタイム表示）
                    self.current_preset_display.config(text=f"{preset_name} (輝度: {avg_luminance:.1f}, {range_display})")

                    # 範囲が変わった場合のみUIパラメータを更新（カメラには適用しない）
                    if new_range != self.current_range:
                        self.current_range = new_range
                        self.current_auto_preset = selected_preset_idx

                        # プリセットのパラメータを取得してUIの値を更新（カメラ設定は変更しない）
                        params = self.presets[selected_preset_idx].get('params', {})
                        if params:
                            # UIの値のみ更新
                            self.param_exposure.set(params.get('exposure', CAMERA_SETTINGS['exposure_time']))
                            self.param_gain.set(params.get('gain', CAMERA_SETTINGS['gain']))
                            self.param_digital_gain.set(params.get('digital_gain', 1.0))
                            self.param_black_level.set(params.get('black_level', 0))
                            self.param_wb_red.set(params.get('wb_red', 1.0))
                            self.param_wb_blue.set(params.get('wb_blue', 1.0))
                            self.param_highlight_comp.set(params.get('highlight_comp', 1.0))
                            self.param_saturation.set(params.get('saturation', 1.0))
                            self.clahe_clip_limit.set(params.get('clahe_clip', DATASET_SETTINGS['clahe_clip_limit']))
                            self.clahe_tile_size.set(params.get('clahe_tile', DATASET_SETTINGS['clahe_tile_size']))
                            print(f"★自動切り替え実行: {preset_name} (平均輝度: {avg_luminance:.1f}, {range_display})")
                        else:
                            print(f"警告: プリセット '{preset_name}' にパラメータが保存されていません")

                # === 左側：オリジナル画像の表示 ===
                original_display = frame.copy()
                # バウンディングボックスを描画
                for box in yolo_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(original_display, (x1, y1), (x2, y2), (0, 255, 255), 3)

                # リサイズして表示
                original_display = resize_for_display(
                    original_display,
                    GUI_SETTINGS['preview_width'] // 2,  # 幅を半分に
                    GUI_SETTINGS['preview_height']
                )
                rgb_original = cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB)
                pil_original = Image.fromarray(rgb_original)
                photo_original = ImageTk.PhotoImage(image=pil_original)
                self.original_preview_label.config(image=photo_original)
                self.original_preview_label.image = photo_original

                # === 右側：プリセット適用後の画像 ===
                # まずソフトウェア処理を適用（ホワイトバランス、ハイライト圧縮、彩度）
                frame_processed = self.apply_software_processing(frame.copy())

                # 次にCLAHE等を適用
                clip_limit = self.clahe_clip_limit.get()
                tile_size = int(self.clahe_tile_size.get())

                # 適応的処理が有効な場合
                if self.adaptive_processing.get():
                    frame_corrected = self.apply_adaptive_processing(frame_processed, clip_limit, tile_size)
                # CLAHEを適用する場合
                elif DATASET_SETTINGS.get('use_clahe', True):
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                    lab = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    frame_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    frame_corrected = frame_processed.copy()

                # 保存用にフレームを保持
                self.current_frame_corrected = frame_corrected

                # バウンディングボックスを描画
                preset_display = frame_corrected.copy()
                for box in yolo_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(preset_display, (x1, y1), (x2, y2), (0, 255, 255), 3)

                # 自動撮影モードの処理（プリセット適用後の画像を保存）
                if self.auto_capture_running:
                    self.auto_capture_process(frame_corrected, yolo_boxes)

                # リサイズして表示
                preset_display = resize_for_display(
                    preset_display,
                    GUI_SETTINGS['preview_width'] // 2,  # 幅を半分に
                    GUI_SETTINGS['preview_height']
                )
                rgb_preset = cv2.cvtColor(preset_display, cv2.COLOR_BGR2RGB)
                pil_preset = Image.fromarray(rgb_preset)
                photo_preset = ImageTk.PhotoImage(image=pil_preset)
                self.preset_preview_label.config(image=photo_preset)
                self.preset_preview_label.image = photo_preset

            time.sleep(0.03)  # 約30fps

    def inspect_once(self):
        """1回検査を実行（開発用・デバッグ用）"""
        if self.is_inspecting:
            return

        self.is_inspecting = True

        # 現在のフレームをキャプチャ
        frame = self.camera.capture_frame()

        if frame is None:
            messagebox.showerror("エラー", "画像の取得に失敗しました")
            self.is_inspecting = False
            return

        # シワ検出実行
        result, count, debug_images = detect_wrinkles(frame)

        # デバッグ表示
        self.show_debug_images(debug_images)

        # 結果を画像に描画
        result_image = draw_result_on_image(frame, result, count)

        # 輪郭変動スコアを取得
        deviation_score = debug_images.get('_contour_deviation', 0.0)

        # カメラ情報を取得
        camera_info = {
            'exposure_time': CAMERA_SETTINGS['exposure_time'],
            'gain': CAMERA_SETTINGS['gain'],
            'brightness': CAMERA_SETTINGS['brightness'],
            'camera_index': self.camera.current_camera_index,
            'backend': self.camera.current_backend
        }

        # 検出パラメータを取得
        detection_params = {
            'binary_threshold': DETECTION_PARAMS['binary_threshold'],
            'wrinkle_line_threshold': DETECTION_PARAMS['wrinkle_line_threshold'],
            'contour_deviation': deviation_score,
            'contour_deviation_threshold': DETECTION_PARAMS.get('contour_deviation_threshold', 10.0)
        }

        # デバッグ画像保存
        debug_path = save_debug_images(debug_images, result, count, camera_info, detection_params)
        if debug_path:
            print(f"デバッグ画像保存: {debug_path}")

        messagebox.showinfo("検査完了", f"結果: {result}\n輪郭変動: {deviation_score:.2f} px")

        self.is_inspecting = False

    def show_debug_images(self, debug_images):
        """
        処理途中経過を別ウィンドウで表示

        Args:
            debug_images: デバッグ画像の辞書
        """
        # 新しいウィンドウを作成
        debug_window = tk.Toplevel(self.root)
        debug_window.title("処理途中経過（クリックで拡大）")

        # ウィンドウサイズを設定
        screen_width = debug_window.winfo_screenwidth()
        screen_height = debug_window.winfo_screenheight()
        window_width = min(700, int(screen_width * 0.5))
        window_height = min(900, int(screen_height * 0.8))
        debug_window.geometry(f"{window_width}x{window_height}")

        # スクロールバー付きキャンバスを作成
        canvas = tk.Canvas(debug_window, bg='white')
        v_scrollbar = ttk.Scrollbar(debug_window, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(debug_window, orient=tk.HORIZONTAL, command=canvas.xview)

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # スクロール可能なフレームを作成
        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)

        # 表示する画像とタイトル
        image_titles = [
            ('1_grayscale', 'グレースケール'),
            ('2_binary', '2値化（参考）'),
            ('3_cleaned', 'ノイズ除去（参考）'),
            ('4_edges', 'エッジ検出（参考）'),
            ('5_detected_lines', '線検出（参考）'),
            ('6_yolo_detection', 'YOLOボトル検出'),
            ('7_bottle_mask', 'ボトル形状抽出'),
            ('8_filled_bottle', 'ボトルベタ塗り'),
            ('9_contour_analysis', '輪郭分析（判定）')
        ]

        # 4行2列のグリッドレイアウト
        for idx, (key, title) in enumerate(image_titles):
            if key in debug_images:
                row = idx // 2
                col = idx % 2

                # フレーム作成
                frame = ttk.LabelFrame(scrollable_frame, text=title, padding="5")
                frame.grid(row=row, column=col, padx=5, pady=5)

                # 画像をリサイズして表示
                img = debug_images[key]

                # グレースケール画像の場合、カラーに変換
                if len(img.shape) == 2:
                    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # リサイズ（小さく表示）
                display_width = 300
                display_height = int(img_display.shape[0] * display_width / img_display.shape[1])
                img_resized = cv2.resize(img_display, (display_width, display_height))

                # PIL Image → ImageTk
                pil_image = Image.fromarray(img_resized)
                photo = ImageTk.PhotoImage(image=pil_image)

                # ラベルに表示
                label = ttk.Label(frame, image=photo, cursor="hand2")
                label.image = photo  # 参照を保持
                label.pack()

                # クリックで拡大表示
                # 元画像を保持（クロージャで使用）
                original_img = img_display.copy()
                label.bind("<Button-1>", lambda e, img=original_img, t=title: self.show_enlarged_image(img, t))

        # 閉じるボタン
        close_button = ttk.Button(scrollable_frame, text="閉じる", command=debug_window.destroy)
        close_button.grid(row=5, column=0, columnspan=2, pady=10)

        # スクロール範囲を更新
        scrollable_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def show_enlarged_image(self, image, title):
        """
        画像を拡大表示（モーダルダイアログ風）

        Args:
            image: 表示する画像（RGB形式）
            title: ウィンドウタイトル
        """
        # 新しいウィンドウを作成（トップレベル）
        enlarge_window = tk.Toplevel(self.root)
        enlarge_window.title(f"拡大表示 - {title}")

        # モーダル風に設定
        enlarge_window.transient(self.root)  # 親ウィンドウに関連付け
        enlarge_window.grab_set()  # モーダル化

        # ウィンドウを最前面に
        enlarge_window.lift()
        enlarge_window.attributes('-topmost', True)
        enlarge_window.after(100, lambda: enlarge_window.attributes('-topmost', False))

        # スクロールバー付きキャンバス
        canvas = tk.Canvas(enlarge_window, bg='gray')
        v_scrollbar = ttk.Scrollbar(enlarge_window, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(enlarge_window, orient=tk.HORIZONTAL, command=canvas.xview)

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 画像を表示
        pil_image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=pil_image)

        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # 参照を保持

        # スクロール範囲を設定
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # ウィンドウサイズを設定（画像サイズに合わせるが、画面の80%まで）
        screen_width = enlarge_window.winfo_screenwidth()
        screen_height = enlarge_window.winfo_screenheight()

        window_width = min(image.shape[1] + 20, int(screen_width * 0.8))
        window_height = min(image.shape[0] + 20, int(screen_height * 0.8))

        enlarge_window.geometry(f"{window_width}x{window_height}")

        # 中央に配置
        enlarge_window.update_idletasks()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        enlarge_window.geometry(f"+{x}+{y}")

        # ESCキーで閉じる
        enlarge_window.bind('<Escape>', lambda e: enlarge_window.destroy())

    def scan_cameras(self):
        """カメラをスキャンして一覧を更新"""
        # カメラが起動中の場合はスキャンしない
        if self.is_running:
            messagebox.showwarning("警告", "カメラを停止してから再検出してください")
            return

        # スキャン実行
        self.available_cameras = self.camera.scan_available_cameras()

        # コンボボックスを更新
        camera_names = [cam['name'] for cam in self.available_cameras]
        self.camera_combo['values'] = camera_names

        if camera_names:
            self.camera_combo.current(0)  # 最初のカメラを選択
            # 起動時はメッセージなし、手動再検出時のみメッセージ表示
            # messagebox.showinfo("スキャン完了", f"{len(camera_names)}台のカメラが見つかりました")
        else:
            messagebox.showwarning("警告", "利用可能なカメラが見つかりませんでした")

    def on_camera_change(self, event):
        """カメラ選択が変更された時"""
        if self.is_running:
            # カメラ起動中の場合は切り替え処理を実行
            selected_index = self.camera_combo.current()
            if selected_index >= 0:
                camera_info = self.available_cameras[selected_index]
                camera_index = camera_info['index']

                # カメラを切り替え
                if self.camera.switch_camera(camera_index):
                    # カメラの基本設定のみ適用（回転とBayerパターン）
                    self.camera.set_rotation(self.rotation_angle.get())
                    # 露出・ゲイン等は固定値を維持（オリジナル画像用）
                else:
                    messagebox.showerror("エラー", "カメラの切り替えに失敗しました")

    def reset_camera_params(self):
        """カメラパラメータをデフォルトに戻す（UIの値のみ）"""
        self.param_exposure.set(CAMERA_SETTINGS['exposure_time'])
        self.param_gain.set(CAMERA_SETTINGS['gain'])
        self.param_digital_gain.set(1.0)
        self.param_black_level.set(0)
        self.param_wb_red.set(1.0)
        self.param_wb_blue.set(1.0)
        self.param_highlight_comp.set(1.0)
        self.param_saturation.set(1.0)
        # Bayerパターンは"BG"固定（変更不可）

        # 注意：カメラには設定を適用しない（オリジナル画像は固定パラメータで取得）
        # UIパラメータは右側のプレビュー（プリセット適用後）に反映される

    def reset_clahe_params(self):
        """CLAHEパラメータをデフォルトに戻す"""
        self.clahe_clip_limit.set(DATASET_SETTINGS['clahe_clip_limit'])
        self.clahe_tile_size.set(DATASET_SETTINGS['clahe_tile_size'])

    def on_rotation_change(self):
        """回転角度変更時のコールバック"""
        if self.is_running:
            self.camera.set_rotation(self.rotation_angle.get())
            print(f"画像回転: {self.rotation_angle.get()}度")

    def apply_software_processing(self, image_bgr):
        """
        ソフトウェア処理を適用（明るさ調整、ホワイトバランス、ハイライト圧縮、彩度）

        Args:
            image_bgr: 入力画像

        Returns:
            処理済み画像
        """
        # 1. 明るさ調整（露出時間・ゲインのシミュレーション）
        current_exposure = self.param_exposure.get()
        current_gain = self.param_gain.get()

        # 露出時間の比率
        exposure_ratio = current_exposure / self.base_exposure

        # ゲインの比率（dBからリニアへの変換）
        gain_diff = current_gain - self.base_gain
        gain_ratio = 10 ** (gain_diff / 20.0)

        # 合計の明るさ倍率
        brightness_multiplier = exposure_ratio * gain_ratio

        # デバッグ: 明るさ倍率を表示（倍率が大きく変わった時のみ）
        if not hasattr(self, '_last_brightness_multiplier'):
            self._last_brightness_multiplier = 1.0

        if abs(brightness_multiplier - self._last_brightness_multiplier) > 0.1:
            print(f"[明るさ調整] 露出比: {exposure_ratio:.2f}x, ゲイン比: {gain_ratio:.2f}x, 合計: {brightness_multiplier:.2f}x")
            self._last_brightness_multiplier = brightness_multiplier

        # デジタルゲインを適用
        if brightness_multiplier != 1.0:
            image_float = image_bgr.astype(np.float32)
            image_float *= brightness_multiplier
            image_bgr = np.clip(image_float, 0, 255).astype(np.uint8)

        # 2. ホワイトバランス
        wb_red = self.param_wb_red.get()
        wb_blue = self.param_wb_blue.get()

        if wb_red != 1.0 or wb_blue != 1.0:
            image_float = image_bgr.astype(np.float32)
            image_float[:, :, 2] *= wb_red   # Rチャンネル
            image_float[:, :, 0] *= wb_blue  # Bチャンネル
            image_bgr = np.clip(image_float, 0, 255).astype(np.uint8)

        # 3. ハイライト圧縮
        highlight_comp = self.param_highlight_comp.get()
        if highlight_comp < 1.0:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)

            # 明るい部分（150以上）を圧縮
            threshold = 150.0
            max_brightness = 180.0
            bright_mask = l_channel > threshold

            range_max = threshold + (max_brightness - threshold) + (255 - max_brightness) * highlight_comp
            compressed = threshold + (l_channel - threshold) * (range_max - threshold) / (255 - threshold)
            l_channel[bright_mask] = compressed[bright_mask]

            lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 4. 彩度調整
        saturation = self.param_saturation.get()
        if saturation < 1.0:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation  # Sチャンネル（彩度）を調整
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image_bgr

    def apply_adaptive_processing(self, image_bgr, clip_limit, tile_size):
        """
        適応的処理：画像内の明暗領域に応じて異なる処理を適用

        Args:
            image_bgr: 入力画像
            clip_limit: CLAHEクリップ限界
            tile_size: CLAHEタイルサイズ

        Returns:
            処理済み画像
        """
        # 1. 輝度マスクを作成
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0]

        # 2. 明るい領域と暗い領域を検出
        bright_mask = (luminance > 150).astype(np.float32)
        dark_mask = (luminance < 100).astype(np.float32)

        # 3. ガウシアンブラーで境界を滑らかに（重み付け平均）
        bright_mask = cv2.GaussianBlur(bright_mask, (51, 51), 0)
        dark_mask = cv2.GaussianBlur(dark_mask, (51, 51), 0)

        # 4. 明るい部分用の処理（ハイライト圧縮強化）
        lab_bright = lab.copy()
        l_channel_bright = lab_bright[:, :, 0].astype(np.float32)
        # 150以上を強く圧縮
        threshold = 150.0
        compressed = threshold + (l_channel_bright - threshold) * 0.3
        lab_bright[:, :, 0] = np.clip(compressed, 0, 255).astype(np.uint8)
        image_bright = cv2.cvtColor(lab_bright, cv2.COLOR_LAB2BGR)

        # 5. 暗い部分用の処理（CLAHE強化）
        lab_dark = lab.copy()
        clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        lab_dark[:, :, 0] = clahe_strong.apply(lab_dark[:, :, 0])
        image_dark = cv2.cvtColor(lab_dark, cv2.COLOR_LAB2BGR)

        # 6. 中間部分（通常のCLAHE）
        lab_mid = lab.copy()
        clahe_mid = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        lab_mid[:, :, 0] = clahe_mid.apply(lab_mid[:, :, 0])
        image_mid = cv2.cvtColor(lab_mid, cv2.COLOR_LAB2BGR)

        # 7. 重み付け合成
        image_float = image_mid.astype(np.float32)
        image_bright_f = image_bright.astype(np.float32)
        image_dark_f = image_dark.astype(np.float32)

        # 明るい部分を合成
        result = image_float * (1 - bright_mask[:, :, np.newaxis]) + \
                 image_bright_f * bright_mask[:, :, np.newaxis]

        # 暗い部分を合成
        result = result * (1 - dark_mask[:, :, np.newaxis]) + \
                 image_dark_f * dark_mask[:, :, np.newaxis]

        result_rgb = result.astype(np.uint8)

        # 8. グレースケール（白黒）に変換
        gray = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2GRAY)

        # BGR形式に戻す（表示用）
        result_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return result_gray_bgr

    def on_camera_param_change(self, value):
        """カメラパラメータ変更時のコールバック（UIの表示のみ更新）"""
        # ラベルを更新
        self.exposure_label.config(text=f"{self.param_exposure.get()}μs")
        self.gain_label.config(text=f"{self.param_gain.get():.1f}dB")
        self.digital_gain_label.config(text=f"{self.param_digital_gain.get():.2f}x")
        self.black_level_label.config(text=f"{self.param_black_level.get()}")
        self.wb_red_label.config(text=f"{self.param_wb_red.get():.2f}")
        self.wb_blue_label.config(text=f"{self.param_wb_blue.get():.2f}")
        self.highlight_comp_label.config(text=f"{self.param_highlight_comp.get():.2f}")
        self.saturation_label.config(text=f"{self.param_saturation.get():.2f}")

        # 注意：カメラには設定を適用しない（オリジナル画像は固定パラメータで取得）
        # UIパラメータは右側のプレビュー（プリセット適用後）に反映される

    def on_interval_change(self, value):
        """撮影間隔変更時"""
        interval = float(value)
        self.capture_interval_label.config(text=f"{interval:.1f}")

    def set_normal_mode(self):
        """通常モードに設定（UIの値のみ）"""
        # スライダーの値を通常モードに変更
        self.param_exposure.set(CAMERA_SETTINGS['exposure_time'])
        self.param_gain.set(CAMERA_SETTINGS['gain'])
        print(f"通常モード（右側プレビューに適用）: 露出={self.param_exposure.get()}μs, ゲイン={self.param_gain.get()}dB")

    def set_fast_mode(self):
        """ブレ防止モードに設定（UIの値のみ）"""
        # スライダーの値をブレ防止モードに変更
        self.param_exposure.set(CAMERA_SETTINGS['exposure_time_fast'])
        self.param_gain.set(CAMERA_SETTINGS['gain_fast'])
        print(f"ブレ防止モード（右側プレビューに適用）: 露出={self.param_exposure.get()}μs, ゲイン={self.param_gain.get()}dB")

    def load_presets(self):
        """プリセットをファイルから読み込み"""
        if os.path.exists(self.preset_file):
            try:
                with open(self.preset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"プリセット読み込みエラー: {e}")

        # デフォルトのプリセット（5つ）
        return [
            {'name': f'プリセット{i+1}', 'params': {}} for i in range(5)
        ]

    def save_presets_to_file(self):
        """プリセットをファイルに保存"""
        try:
            with open(self.preset_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("エラー", f"プリセット保存失敗: {e}")

    def get_current_params(self):
        """現在のパラメータを取得"""
        return {
            'exposure': self.param_exposure.get(),
            'gain': self.param_gain.get(),
            'digital_gain': self.param_digital_gain.get(),
            'black_level': self.param_black_level.get(),
            'wb_red': self.param_wb_red.get(),
            'wb_blue': self.param_wb_blue.get(),
            'highlight_comp': self.param_highlight_comp.get(),
            'saturation': self.param_saturation.get(),
            'clahe_clip': self.clahe_clip_limit.get(),
            'clahe_tile': self.clahe_tile_size.get()
        }

    def apply_params(self, params):
        """パラメータを適用（UIの値のみ更新、カメラには適用しない）"""
        self.param_exposure.set(params.get('exposure', CAMERA_SETTINGS['exposure_time']))
        self.param_gain.set(params.get('gain', CAMERA_SETTINGS['gain']))
        self.param_digital_gain.set(params.get('digital_gain', 1.0))
        self.param_black_level.set(params.get('black_level', 0))
        self.param_wb_red.set(params.get('wb_red', 1.0))
        self.param_wb_blue.set(params.get('wb_blue', 1.0))
        self.param_highlight_comp.set(params.get('highlight_comp', 1.0))
        self.param_saturation.set(params.get('saturation', 1.0))
        self.clahe_clip_limit.set(params.get('clahe_clip', DATASET_SETTINGS['clahe_clip_limit']))
        self.clahe_tile_size.set(params.get('clahe_tile', DATASET_SETTINGS['clahe_tile_size']))

        # 注意：カメラには設定を適用しない（オリジナル画像は固定パラメータで取得）
        # UIパラメータは右側のプレビュー（プリセット適用後）に反映される

    def save_preset(self, index):
        """プリセットを保存"""
        self.presets[index]['params'] = self.get_current_params()
        self.save_presets_to_file()
        messagebox.showinfo("保存完了", f"'{self.presets[index]['name']}' に現在の設定を保存しました")

    def load_preset(self, index):
        """プリセットを読み込み"""
        params = self.presets[index].get('params', {})
        if not params:
            messagebox.showwarning("警告", f"'{self.presets[index]['name']}' には設定が保存されていません")
            return

        self.apply_params(params)
        print(f"プリセット '{self.presets[index]['name']}' を読み込みました")

    def rename_preset(self, index):
        """プリセット名を変更"""
        current_name = self.presets[index]['name']
        new_name = simpledialog.askstring("プリセット名変更",
                                         f"新しい名前を入力してください:",
                                         initialvalue=current_name)
        if new_name:
            self.presets[index]['name'] = new_name
            self.preset_labels[index].config(text=new_name)
            self.save_presets_to_file()
            print(f"プリセット名を '{new_name}' に変更しました")

    def update_auto_preset_labels(self, *args):
        """自動プリセット選択ドロップダウンのラベルを更新"""
        idx1 = self.preset_range1.get()
        idx2 = self.preset_range2.get()
        idx3 = self.preset_range3.get()
        idx4 = self.preset_range4.get()

        name1 = self.presets[idx1].get('name', f'プリセット{idx1+1}')
        name2 = self.presets[idx2].get('name', f'プリセット{idx2+1}')
        name3 = self.presets[idx3].get('name', f'プリセット{idx3+1}')
        name4 = self.presets[idx4].get('name', f'プリセット{idx4+1}')

        self.preset_range1_label.config(text=name1)
        self.preset_range2_label.config(text=name2)
        self.preset_range3_label.config(text=name3)
        self.preset_range4_label.config(text=name4)

    def load_auto_preset_config(self):
        """自動プリセット設定を読み込み"""
        if os.path.exists(self.auto_preset_config_file):
            try:
                with open(self.auto_preset_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.threshold1.set(config.get('threshold1', 60))
                    self.threshold2.set(config.get('threshold2', 100))
                    self.threshold3.set(config.get('threshold3', 140))
                    self.preset_range1.set(config.get('preset_range1', 0))
                    self.preset_range2.set(config.get('preset_range2', 1))
                    self.preset_range3.set(config.get('preset_range3', 2))
                    self.preset_range4.set(config.get('preset_range4', 3))
                    self.luminance_hysteresis.set(config.get('hysteresis', 10))
                    print(f"自動プリセット設定を読み込みました")
            except Exception as e:
                print(f"自動プリセット設定読み込みエラー: {e}")

    def save_auto_preset_config(self):
        """自動プリセット設定を保存"""
        try:
            config = {
                'threshold1': self.threshold1.get(),
                'threshold2': self.threshold2.get(),
                'threshold3': self.threshold3.get(),
                'preset_range1': self.preset_range1.get(),
                'preset_range2': self.preset_range2.get(),
                'preset_range3': self.preset_range3.get(),
                'preset_range4': self.preset_range4.get(),
                'hysteresis': self.luminance_hysteresis.get()
            }
            with open(self.auto_preset_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存完了", "自動プリセット設定を保存しました")
        except Exception as e:
            messagebox.showerror("エラー", f"設定保存失敗: {e}")

    def start_auto_capture(self):
        """自動撮影を開始"""
        self.auto_capture_running = True
        self.last_capture_time = 0
        self.bottle_detected = False

        # ボタン状態変更
        self.auto_start_button.config(state=tk.DISABLED)
        self.auto_stop_button.config(state=tk.NORMAL)
        self.collect_ok_button.config(state=tk.DISABLED)
        self.collect_ng_button.config(state=tk.DISABLED)

        self.auto_status_label.config(text="自動撮影中 - ボトル待機中", foreground="blue")
        # messagebox.showinfo("開始", "自動撮影を開始しました\nボトルを検出すると自動的に撮影します")

    def stop_auto_capture(self):
        """自動撮影を停止"""
        self.auto_capture_running = False

        # ボタン状態変更
        self.auto_start_button.config(state=tk.NORMAL)
        self.auto_stop_button.config(state=tk.DISABLED)
        self.collect_ok_button.config(state=tk.NORMAL)
        self.collect_ng_button.config(state=tk.NORMAL)

        self.auto_status_label.config(text="待機中", foreground="black")

    def auto_capture_process(self, frame, yolo_boxes):
        """
        自動撮影の処理（プレビューループから呼ばれる）

        Args:
            frame: 現在のフレーム
            yolo_boxes: YOLOで検出したボトルの範囲
        """
        try:
            current_time = time.time()
            interval = self.capture_interval_var.get()

            if yolo_boxes and len(yolo_boxes) > 0:
                # ボトル検出中
                if not self.bottle_detected:
                    # 新しいボトルを検出
                    self.bottle_detected = True
                    self.last_capture_time = 0  # リセット
                    self.auto_status_label.config(text="ボトル検出 - 撮影中", foreground="green")

                # 一定間隔で撮影
                if current_time - self.last_capture_time >= interval:
                    # OK品として自動保存
                    save_dataset_image(frame, "ok")
                    self.last_capture_time = current_time

                    # カウンター更新
                    self.update_dataset_count()

                    # 撮影インジケーターを表示（フラッシュ効果）
                    self.show_capture_flash()

                    print(f"自動撮影: OK品を保存しました")

            else:
                # ボトルが検出されていない
                if self.bottle_detected:
                    # ボトルが画面から消えた
                    self.bottle_detected = False
                    self.auto_status_label.config(text="自動撮影中 - 次のボトル待機中", foreground="blue")

        except Exception as e:
            print(f"自動撮影エラー: {e}")

    def save_to_dataset(self, label):
        """
        現在のフレームを学習データセットとして保存（CLAHE適用済み）

        Args:
            label: "ok" or "ng"
        """
        # プレビュー中のCLAHE適用済みフレームを使用
        if self.current_frame_corrected is None:
            messagebox.showerror("エラー", "画像の取得に失敗しました")
            return

        # データセットに保存
        try:
            filepath = save_dataset_image(self.current_frame_corrected, label)
            # messagebox.showinfo("成功", f"{label.upper()}品として保存しました\n{filepath}")

            # カウンターを更新
            self.update_dataset_count()

        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました: {e}")

    def show_capture_flash(self):
        """撮影時のフラッシュ効果を表示"""
        # インジケーターを緑色で表示
        self.capture_indicator.config(text="📸 撮影!", foreground="white", background="green")

        # 0.3秒後に元に戻す
        def reset_indicator():
            self.capture_indicator.config(text="", background="white")

        self.root.after(300, reset_indicator)

    def update_dataset_count(self):
        """データ収集状況を更新"""
        try:
            ok_count, ng_count = get_dataset_count()
            target_ok = DATASET_SETTINGS['target_ok_count']
            target_ng = DATASET_SETTINGS['target_ng_count']

            self.dataset_label.config(text=f"OK: {ok_count}/{target_ok} | NG: {ng_count}/{target_ng}")

            # 目標達成チェック
            if ok_count >= target_ok and ng_count >= target_ng:
                self.dataset_label.config(foreground="green")
            else:
                self.dataset_label.config(foreground="black")

        except Exception as e:
            self.dataset_label.config(text=f"エラー: {e}")

    def open_review_window(self):
        """OK画像レビューウィンドウを開く"""
        import glob
        from config import DATASET_SETTINGS

        # OK画像のリストを取得
        ok_images = glob.glob(f"{DATASET_SETTINGS['ok_dir']}/*.jpg")

        if not ok_images:
            messagebox.showinfo("情報", "レビューするOK画像がありません")
            return

        # レビューウィンドウを作成
        review_window = tk.Toplevel(self.root)
        review_window.title(f"OK画像レビュー - {len(ok_images)}枚")
        review_window.geometry("800x700")

        current_index = [0]  # リストで保持（クロージャで変更可能に）

        # 画像表示エリア
        image_label = ttk.Label(review_window)
        image_label.pack(pady=10)

        # 画像情報
        info_label = ttk.Label(review_window, text="", font=('Arial', 10))
        info_label.pack()

        # ボタンフレーム
        button_frame = ttk.Frame(review_window)
        button_frame.pack(pady=10)

        def show_image():
            """現在の画像を表示"""
            if 0 <= current_index[0] < len(ok_images):
                img_path = ok_images[current_index[0]]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # リサイズ
                img = cv2.resize(img, (640, 480))

                pil_img = Image.fromarray(img)
                photo = ImageTk.PhotoImage(image=pil_img)

                image_label.config(image=photo)
                image_label.image = photo

                info_label.config(text=f"{current_index[0] + 1}/{len(ok_images)} - {os.path.basename(img_path)}")

        def next_image():
            """次の画像"""
            if current_index[0] < len(ok_images) - 1:
                current_index[0] += 1
                show_image()

        def prev_image():
            """前の画像"""
            if current_index[0] > 0:
                current_index[0] -= 1
                show_image()

        def move_to_ng():
            """現在の画像をNGフォルダに移動"""
            if 0 <= current_index[0] < len(ok_images):
                src = ok_images[current_index[0]]
                dst = src.replace(DATASET_SETTINGS['ok_dir'], DATASET_SETTINGS['ng_dir'])

                # ファイル移動
                import shutil
                shutil.move(src, dst)

                messagebox.showinfo("移動完了", f"NG品として移動しました")

                # リストから削除
                ok_images.pop(current_index[0])

                # カウンター更新
                self.update_dataset_count()

                # 次の画像を表示
                if len(ok_images) == 0:
                    review_window.destroy()
                    messagebox.showinfo("完了", "すべての画像をレビューしました")
                else:
                    if current_index[0] >= len(ok_images):
                        current_index[0] = len(ok_images) - 1
                    show_image()

        def annotate_wrinkle():
            """シワをアノテーション"""
            if 0 <= current_index[0] < len(ok_images):
                from annotation_tool import WrinkleAnnotationTool

                def on_annotation_saved(image_path, polygons):
                    print(f"アノテーション保存完了: {len(polygons)}個のシワ")

                # アノテーションツールを開く
                WrinkleAnnotationTool(review_window, ok_images[current_index[0]], on_annotation_saved)

        # ボタン配置
        ttk.Button(button_frame, text="← 前へ", command=prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="シワをアノテーション", command=annotate_wrinkle).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="NG品として移動", command=move_to_ng).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="次へ →", command=next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="閉じる", command=review_window.destroy).pack(side=tk.LEFT, padx=5)

        # 最初の画像を表示
        show_image()

    def on_closing(self):
        """ウィンドウを閉じる時の処理"""
        # カメラが起動中の場合は停止
        if self.is_running:
            self.is_running = False
            time.sleep(0.5)  # スレッド終了を待つ
            self.camera.close()

        # ウィンドウを閉じる
        self.root.destroy()

    def run(self):
        """アプリケーション実行"""
        # mainloopの代わりに、定期的にチェックするループを使用
        def check_quit():
            try:
                self.root.after(100, check_quit)
            except:
                pass

        check_quit()
        self.root.mainloop()


def main():
    """メイン関数"""
    root = tk.Tk()
    app = WrinkleDetectionApp(root)

    # クリーンアップ関数
    def cleanup():
        print("\nカメラをクローズしています...")
        try:
            if app.is_running:
                app.is_running = False
                time.sleep(0.5)
            app.camera.close()
        except Exception as e:
            print(f"クリーンアップエラー: {e}")
        try:
            root.quit()
            root.destroy()
        except:
            pass

    # Ctrl+Cハンドラ
    import signal
    def signal_handler(sig, frame):
        print("\n強制終了シグナルを受信しました...")
        cleanup()
        import sys
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        app.run()
    except KeyboardInterrupt:
        cleanup()
        print("正常終了しました")
    except Exception as e:
        cleanup()
        print(f"エラーで終了: {e}")
    finally:
        try:
            cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
