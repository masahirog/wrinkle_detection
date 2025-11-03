# -*- coding: utf-8 -*-
"""
メインプログラム
シャンパンボトル シワ検査システム Phase 1
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import signal
import sys

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

        # 現在のフレーム（CLAHE適用済み）
        self.current_frame_corrected = None

        # CLAHEパラメータ
        self.clahe_clip_limit = tk.DoubleVar(value=DATASET_SETTINGS['clahe_clip_limit'])
        self.clahe_tile_size = tk.IntVar(value=DATASET_SETTINGS['clahe_tile_size'])

        # 統計情報
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0

        # 利用可能なカメラリスト
        self.available_cameras = []

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

        # 左側: カメラプレビュー
        preview_frame = ttk.LabelFrame(main_frame, text="カメラプレビュー", padding="10")
        preview_frame.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky=(tk.N, tk.S))

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack()

        # 右上: カメラ制御
        camera_control_frame = ttk.LabelFrame(main_frame, text="カメラ制御", padding="10")
        camera_control_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

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

        # モード切り替えボタン
        mode_button_frame = ttk.Frame(camera_control_frame)
        mode_button_frame.grid(row=1, column=0, columnspan=4, pady=10)

        ttk.Button(mode_button_frame, text="通常モード", command=self.set_normal_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(mode_button_frame, text="ブレ防止モード", command=self.set_fast_mode).pack(side=tk.LEFT, padx=5)

        # カメラ起動/停止ボタン
        self.start_button = ttk.Button(camera_control_frame, text="カメラ起動", command=self.start_camera)
        self.start_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.stop_button = ttk.Button(camera_control_frame, text="カメラ停止", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # CLAHE調整スライダー
        clahe_frame = ttk.LabelFrame(camera_control_frame, text="CLAHE調整（白飛び・黒つぶれ対策）", padding="5")
        clahe_frame.grid(row=3, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(clahe_frame, text="クリップ限界:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.clahe_clip_scale = ttk.Scale(clahe_frame, from_=0.5, to=4.0,
                                         variable=self.clahe_clip_limit,
                                         orient=tk.HORIZONTAL, length=150)
        self.clahe_clip_scale.grid(row=0, column=1, padx=5, pady=2)
        self.clahe_clip_label = ttk.Label(clahe_frame, text=f"{self.clahe_clip_limit.get():.1f}")
        self.clahe_clip_label.grid(row=0, column=2, padx=5)
        self.clahe_clip_limit.trace_add('write', lambda *args: self.clahe_clip_label.config(text=f"{self.clahe_clip_limit.get():.1f}"))

        ttk.Label(clahe_frame, text="タイルサイズ:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.clahe_tile_scale = ttk.Scale(clahe_frame, from_=2, to=16,
                                         variable=self.clahe_tile_size,
                                         orient=tk.HORIZONTAL, length=150)
        self.clahe_tile_scale.grid(row=1, column=1, padx=5, pady=2)
        self.clahe_tile_label = ttk.Label(clahe_frame, text=f"{self.clahe_tile_size.get()}")
        self.clahe_tile_label.grid(row=1, column=2, padx=5)
        self.clahe_tile_size.trace_add('write', lambda *args: self.clahe_tile_label.config(text=f"{self.clahe_tile_size.get()}"))

        # データ収集フレーム
        control_frame = ttk.LabelFrame(main_frame, text="データ収集", padding="10")
        control_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))

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

            # カメラ設定を適用（デフォルト値）
            self.camera.set_exposure(CAMERA_SETTINGS['exposure_time'])
            self.camera.set_gain(CAMERA_SETTINGS['gain'])
            self.camera.set_brightness(CAMERA_SETTINGS['brightness'])

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
        """プレビューループ（別スレッド）"""
        while self.is_running:
            frame = self.camera.capture_frame()

            if frame is not None:
                # CLAHE（適応的ヒストグラム平坦化）を適用
                # 白いラベルと黒いラベルの両方でシワが見えるように補正
                # スライダーの値を使って動的に調整
                clip_limit = self.clahe_clip_limit.get()
                tile_size = int(self.clahe_tile_size.get())

                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

                # L*a*b*色空間でCLAHE適用
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # 現在のフレームを保存（手動保存時に使用）
                self.current_frame_corrected = frame_corrected

                # YOLOでボトル検出（CLAHE適用後の画像で）
                try:
                    yolo_boxes, display_frame = detect_bottle_with_yolo(frame_corrected)

                    # 自動撮影モードの処理（CLAHE適用後の画像を保存）
                    if self.auto_capture_running:
                        self.auto_capture_process(frame_corrected, yolo_boxes)
                except:
                    # YOLO失敗時は補正済み画像を使用
                    display_frame = frame_corrected

                # リサイズして表示
                display_frame = resize_for_display(
                    display_frame,
                    GUI_SETTINGS['preview_width'],
                    GUI_SETTINGS['preview_height']
                )

                # BGR → RGB変換
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # PIL Image → ImageTk
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=pil_image)

                # ラベルに表示
                self.preview_label.config(image=photo)
                self.preview_label.image = photo  # 参照を保持

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
                    # カメラ設定を再適用（デフォルト値）
                    self.camera.set_exposure(CAMERA_SETTINGS['exposure_time'])
                    self.camera.set_gain(CAMERA_SETTINGS['gain'])
                    self.camera.set_brightness(CAMERA_SETTINGS['brightness'])
                    # messagebox.showinfo("成功", f"カメラ {camera_index} に切り替えました")
                else:
                    messagebox.showerror("エラー", "カメラの切り替えに失敗しました")

    def on_interval_change(self, value):
        """撮影間隔変更時"""
        interval = float(value)
        self.capture_interval_label.config(text=f"{interval:.1f}")

    def set_normal_mode(self):
        """通常モードに設定"""
        if self.is_running:
            self.camera.set_exposure(CAMERA_SETTINGS['exposure_time'])
            self.camera.set_gain(CAMERA_SETTINGS['gain'])
            self.camera.set_brightness(CAMERA_SETTINGS['brightness'])
            print(f"通常モード: 露出={CAMERA_SETTINGS['exposure_time']}μs, ゲイン={CAMERA_SETTINGS['gain']}dB")

    def set_fast_mode(self):
        """ブレ防止モードに設定"""
        if self.is_running:
            self.camera.set_exposure(CAMERA_SETTINGS['exposure_time_fast'])
            self.camera.set_gain(CAMERA_SETTINGS['gain_fast'])
            self.camera.set_brightness(CAMERA_SETTINGS['brightness_fast'])
            print(f"ブレ防止モード: 露出={CAMERA_SETTINGS['exposure_time_fast']}μs, ゲイン={CAMERA_SETTINGS['gain_fast']}dB")

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
