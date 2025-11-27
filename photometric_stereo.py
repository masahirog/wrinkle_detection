# -*- coding: utf-8 -*-
"""
フォトメトリックステレオ実験ツール

3方向からの照明で撮影した画像から法線マップを計算し、
表面の3D形状を推定する。

使用方法:
1. LED1を点灯して「LED1撮影」ボタンを押す
2. LED2を点灯して「LED2撮影」ボタンを押す
3. LED3を点灯して「LED3撮影」ボタンを押す
4. 「法線マップ計算」ボタンを押す

光源キャリブレーション:
- 光沢のある球（パチンコ玉など）を使って光源方向を自動推定
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import logging

from camera_control import StCameraControl
from config import CAMERA_SETTINGS

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightCalibrationWindow:
    """光源キャリブレーション用ウィンドウ"""

    def __init__(self, parent, image, led_index, callback):
        """
        Args:
            parent: 親ウィンドウ
            image: キャリブレーション用画像（グレースケール）
            led_index: LED番号（0, 1, 2）
            callback: キャリブレーション完了時のコールバック (azimuth, elevation)
        """
        self.window = tk.Toplevel(parent)
        self.window.title(f"光源キャリブレーション - LED{led_index + 1}")
        self.window.geometry("900x700")

        self.image = image
        self.led_index = led_index
        self.callback = callback

        # 球の中心と半径
        self.sphere_center = None
        self.sphere_radius = None

        # マウス操作用
        self.drag_start = None
        self.temp_circle = None

        # 検出結果
        self.highlight_pos = None
        self.light_vector = None

        self._setup_gui()

    def _setup_gui(self):
        """GUI構築"""
        # 説明
        info_frame = ttk.Frame(self.window, padding="10")
        info_frame.pack(fill=tk.X)

        ttk.Label(info_frame, text="操作方法:").pack(anchor=tk.W)
        ttk.Label(info_frame, text="1. 球の中心をクリックし、ドラッグして球の輪郭まで引く").pack(anchor=tk.W)
        ttk.Label(info_frame, text="2. 「ハイライト検出」で光源方向を自動計算").pack(anchor=tk.W)
        ttk.Label(info_frame, text="3. 「適用」で設定を反映").pack(anchor=tk.W)

        # 画像表示
        canvas_frame = ttk.Frame(self.window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # 表示サイズを計算（最大800x500）
        h, w = self.image.shape
        scale = min(800 / w, 500 / h, 1.0)
        self.display_w = int(w * scale)
        self.display_h = int(h * scale)
        self.scale = scale

        self.canvas = tk.Canvas(canvas_frame, width=self.display_w, height=self.display_h, bg='black')
        self.canvas.pack()

        # 画像を表示
        display_img = cv2.resize(self.image, (self.display_w, self.display_h))
        display_rgb = cv2.cvtColor(display_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # マウスイベント
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # 結果表示
        result_frame = ttk.LabelFrame(self.window, text="検出結果", padding="10")
        result_frame.pack(fill=tk.X, padx=10, pady=5)

        self.result_label = ttk.Label(result_frame, text="球を指定してください")
        self.result_label.pack()

        # ボタン
        btn_frame = ttk.Frame(self.window, padding="10")
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="ハイライト検出", command=self._detect_highlight).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="適用", command=self._apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=self.window.destroy).pack(side=tk.LEFT, padx=5)

    def _on_mouse_down(self, event):
        """マウスダウン"""
        self.drag_start = (event.x, event.y)
        if self.temp_circle:
            self.canvas.delete(self.temp_circle)
            self.temp_circle = None

    def _on_mouse_drag(self, event):
        """マウスドラッグ"""
        if self.drag_start:
            if self.temp_circle:
                self.canvas.delete(self.temp_circle)

            cx, cy = self.drag_start
            r = np.sqrt((event.x - cx)**2 + (event.y - cy)**2)

            self.temp_circle = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline='green', width=2
            )

    def _on_mouse_up(self, event):
        """マウスアップ"""
        if self.drag_start:
            cx, cy = self.drag_start
            r = np.sqrt((event.x - cx)**2 + (event.y - cy)**2)

            if r > 10:  # 最小半径
                # 表示座標から元画像座標に変換
                self.sphere_center = (int(cx / self.scale), int(cy / self.scale))
                self.sphere_radius = int(r / self.scale)

                self.result_label.config(
                    text=f"球: 中心=({self.sphere_center[0]}, {self.sphere_center[1]}), 半径={self.sphere_radius}px"
                )

            self.drag_start = None

    def _detect_highlight(self):
        """ハイライト検出"""
        if self.sphere_center is None or self.sphere_radius is None:
            messagebox.showerror("エラー", "先に球を指定してください")
            return

        cx, cy = self.sphere_center
        r = self.sphere_radius

        # 球の領域をマスク
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        # マスク内で最も明るい点を検出
        masked_img = self.image.copy()
        masked_img[mask == 0] = 0

        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(masked_img, (5, 5), 0)

        # 最大輝度の位置を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred, mask=mask)
        self.highlight_pos = max_loc  # (x, y)

        # 光源方向ベクトルを計算
        hx, hy = self.highlight_pos

        # 正規化座標（球の中心を原点、半径を1とする）
        nx = (hx - cx) / r
        ny = (hy - cy) / r

        # 球面上のZ座標
        nz_sq = 1 - nx**2 - ny**2
        if nz_sq < 0:
            nz_sq = 0
        nz = np.sqrt(nz_sq)

        # 鏡面反射を考慮した光源方向
        # 視線方向 V = (0, 0, 1)
        # 法線 N = (nx, ny, nz)
        # 光源方向 L = 2 * (N・V) * N - V = 2*nz*N - V
        # L = (2*nz*nx, 2*nz*ny, 2*nz*nz - 1)
        lx = 2 * nz * nx
        ly = 2 * nz * ny
        lz = 2 * nz * nz - 1

        # 正規化
        length = np.sqrt(lx**2 + ly**2 + lz**2)
        if length > 0:
            lx, ly, lz = lx/length, ly/length, lz/length

        self.light_vector = (lx, ly, lz)

        # 方位角と仰角に変換
        azimuth = np.degrees(np.arctan2(ly, lx))
        elevation = np.degrees(np.arcsin(lz))

        # 結果表示
        self.result_label.config(
            text=f"ハイライト位置: ({hx}, {hy})\n"
                 f"光源ベクトル: ({lx:.3f}, {ly:.3f}, {lz:.3f})\n"
                 f"方位角: {azimuth:.1f}°, 仰角: {elevation:.1f}°"
        )

        # ハイライト位置を表示
        hx_disp = int(hx * self.scale)
        hy_disp = int(hy * self.scale)
        self.canvas.create_oval(
            hx_disp - 5, hy_disp - 5, hx_disp + 5, hy_disp + 5,
            fill='red', outline='yellow', width=2
        )

        logger.info(f"LED{self.led_index + 1} 光源方向: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")

    def _apply(self):
        """適用"""
        if self.light_vector is None:
            messagebox.showerror("エラー", "先にハイライト検出を行ってください")
            return

        lx, ly, lz = self.light_vector

        # 方位角と仰角に変換
        azimuth = np.degrees(np.arctan2(ly, lx))
        elevation = np.degrees(np.arcsin(lz))

        # コールバック呼び出し
        self.callback(azimuth, elevation)
        self.window.destroy()


class PhotometricStereoApp:
    """フォトメトリックステレオGUIアプリケーション"""

    def __init__(self, root):
        self.root = root
        self.root.title("Photometric Stereo - 法線マップ生成")
        self.root.geometry("1400x900")

        # カメラ制御
        self.camera = StCameraControl()
        self.camera_opened = False

        # カメラ回転
        self.rotation = 0

        # 撮影画像（3枚）
        self.captured_images = [None, None, None]

        # 光源方向ベクトル（デフォルト: 3方向から120度間隔、仰角45度）
        # 各光源の方位角（azimuth）と仰角（elevation）
        self.light_params = [
            {'azimuth': 0, 'elevation': 45},     # LED1: 正面
            {'azimuth': 120, 'elevation': 45},   # LED2: 120度
            {'azimuth': 240, 'elevation': 45},   # LED3: 240度
        ]

        # 結果画像
        self.normal_map = None
        self.albedo_map = None

        # 出力ディレクトリ
        self.output_dir = "photometric_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # GUI構築
        self._setup_gui()

        # カメラ起動
        self._start_camera()

        # プレビュー更新開始
        self._update_preview()

    def _setup_gui(self):
        """GUIレイアウト構築"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左側: プレビューと撮影画像
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # カメラプレビュー
        preview_label = ttk.Label(left_frame, text="カメラプレビュー")
        preview_label.pack()

        self.preview_canvas = tk.Canvas(left_frame, width=640, height=400, bg='black')
        self.preview_canvas.pack(pady=5)

        # 撮影画像表示エリア
        captured_frame = ttk.LabelFrame(left_frame, text="撮影画像", padding="5")
        captured_frame.pack(fill=tk.X, pady=10)

        self.captured_canvases = []
        for i in range(3):
            frame = ttk.Frame(captured_frame)
            frame.pack(side=tk.LEFT, padx=5)
            label = ttk.Label(frame, text=f"LED{i+1}")
            label.pack()
            canvas = tk.Canvas(frame, width=200, height=125, bg='gray')
            canvas.pack()
            self.captured_canvases.append(canvas)

        # 右側: 設定と結果
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        # カメラ設定
        camera_frame = ttk.LabelFrame(right_frame, text="カメラ設定", padding="10")
        camera_frame.pack(fill=tk.X, pady=5)

        # 回転設定
        rotation_frame = ttk.Frame(camera_frame)
        rotation_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rotation_frame, text="回転:").pack(side=tk.LEFT)
        self.rotation_var = tk.StringVar(value="0")
        rotation_combo = ttk.Combobox(rotation_frame, textvariable=self.rotation_var,
                                       values=["0", "90", "180", "270"], width=5, state="readonly")
        rotation_combo.pack(side=tk.LEFT, padx=5)
        rotation_combo.bind("<<ComboboxSelected>>", self._on_rotation_change)
        ttk.Label(rotation_frame, text="°").pack(side=tk.LEFT)

        # 露出
        ttk.Label(camera_frame, text="露出時間 (us):").pack(anchor=tk.W)
        self.exposure_var = tk.IntVar(value=CAMERA_SETTINGS['exposure_time'])
        exposure_scale = ttk.Scale(camera_frame, from_=1000, to=100000,
                                   variable=self.exposure_var, orient=tk.HORIZONTAL,
                                   command=self._on_exposure_change)
        exposure_scale.pack(fill=tk.X)
        self.exposure_label = ttk.Label(camera_frame, text=f"{self.exposure_var.get()} us")
        self.exposure_label.pack()

        # ゲイン
        ttk.Label(camera_frame, text="ゲイン (dB):").pack(anchor=tk.W)
        self.gain_var = tk.DoubleVar(value=CAMERA_SETTINGS['gain'])
        gain_scale = ttk.Scale(camera_frame, from_=0, to=30,
                               variable=self.gain_var, orient=tk.HORIZONTAL,
                               command=self._on_gain_change)
        gain_scale.pack(fill=tk.X)
        self.gain_label = ttk.Label(camera_frame, text=f"{self.gain_var.get():.1f} dB")
        self.gain_label.pack()

        # 光源設定
        light_frame = ttk.LabelFrame(right_frame, text="光源方向設定", padding="10")
        light_frame.pack(fill=tk.X, pady=5)

        self.light_vars = []
        for i in range(3):
            led_frame = ttk.Frame(light_frame)
            led_frame.pack(fill=tk.X, pady=2)

            ttk.Label(led_frame, text=f"LED{i+1}:", width=5).pack(side=tk.LEFT)

            # 方位角
            ttk.Label(led_frame, text="方位角:").pack(side=tk.LEFT)
            az_var = tk.DoubleVar(value=self.light_params[i]['azimuth'])
            az_entry = ttk.Entry(led_frame, textvariable=az_var, width=6)
            az_entry.pack(side=tk.LEFT)
            ttk.Label(led_frame, text="°").pack(side=tk.LEFT)

            # 仰角
            ttk.Label(led_frame, text=" 仰角:").pack(side=tk.LEFT)
            el_var = tk.DoubleVar(value=self.light_params[i]['elevation'])
            el_entry = ttk.Entry(led_frame, textvariable=el_var, width=6)
            el_entry.pack(side=tk.LEFT)
            ttk.Label(led_frame, text="°").pack(side=tk.LEFT)

            self.light_vars.append({'azimuth': az_var, 'elevation': el_var})

        # 光源キャリブレーションボタン
        calib_frame = ttk.LabelFrame(right_frame, text="光源キャリブレーション（球を使用）", padding="10")
        calib_frame.pack(fill=tk.X, pady=5)

        ttk.Label(calib_frame, text="LEDを点灯して球を撮影し、光源方向を自動推定").pack(anchor=tk.W)

        calib_btn_frame = ttk.Frame(calib_frame)
        calib_btn_frame.pack(pady=5)

        for i in range(3):
            btn = ttk.Button(calib_btn_frame, text=f"LED{i+1}キャリブ",
                           command=lambda idx=i: self._start_calibration(idx))
            btn.pack(side=tk.LEFT, padx=3)

        # 撮影ボタン
        capture_frame = ttk.LabelFrame(right_frame, text="撮影", padding="10")
        capture_frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(capture_frame)
        btn_frame.pack()

        for i in range(3):
            btn = ttk.Button(btn_frame, text=f"LED{i+1}撮影",
                           command=lambda idx=i: self._capture_image(idx))
            btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(capture_frame, text="全クリア", command=self._clear_all).pack(pady=5)

        # 計算ボタン
        calc_frame = ttk.LabelFrame(right_frame, text="計算", padding="10")
        calc_frame.pack(fill=tk.X, pady=5)

        ttk.Button(calc_frame, text="法線マップ計算",
                  command=self._calculate_normal_map).pack(fill=tk.X, pady=2)
        ttk.Button(calc_frame, text="結果を保存",
                  command=self._save_results).pack(fill=tk.X, pady=2)

        # 結果表示
        result_frame = ttk.LabelFrame(right_frame, text="結果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        result_inner = ttk.Frame(result_frame)
        result_inner.pack()

        # 法線マップ
        normal_frame = ttk.Frame(result_inner)
        normal_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(normal_frame, text="法線マップ").pack()
        self.normal_canvas = tk.Canvas(normal_frame, width=180, height=112, bg='gray')
        self.normal_canvas.pack()

        # アルベド
        albedo_frame = ttk.Frame(result_inner)
        albedo_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(albedo_frame, text="アルベド").pack()
        self.albedo_canvas = tk.Canvas(albedo_frame, width=180, height=112, bg='gray')
        self.albedo_canvas.pack()

        # ステータスバー
        self.status_var = tk.StringVar(value="カメラ起動中...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _start_camera(self):
        """カメラ起動"""
        if self.camera.open():
            self.camera_opened = True
            self.status_var.set("カメラ接続完了")
            logger.info("カメラ起動成功")
        else:
            self.status_var.set("カメラ接続失敗")
            logger.error("カメラ起動失敗")

    def _on_rotation_change(self, event):
        """回転変更"""
        self.rotation = int(self.rotation_var.get())
        if self.camera_opened:
            self.camera.set_rotation(self.rotation)

        # キャンバスサイズを回転に応じて変更
        if self.rotation in [90, 270]:
            # 縦長
            self.preview_canvas.config(width=400, height=640)
        else:
            # 横長
            self.preview_canvas.config(width=640, height=400)

        logger.info(f"回転設定: {self.rotation}°")

    def _on_exposure_change(self, value):
        """露出変更"""
        exposure = int(float(value))
        self.exposure_label.config(text=f"{exposure} us")
        if self.camera_opened:
            self.camera.set_exposure(exposure)

    def _on_gain_change(self, value):
        """ゲイン変更"""
        gain = float(value)
        self.gain_label.config(text=f"{gain:.1f} dB")
        if self.camera_opened:
            self.camera.set_gain(gain)

    def _update_preview(self):
        """プレビュー更新"""
        if self.camera_opened:
            frame = self.camera.capture_frame()
            if frame is not None:
                # グレースケールに変換して表示
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                # 回転に応じてリサイズ
                if self.rotation in [90, 270]:
                    display = cv2.resize(display, (400, 640))
                else:
                    display = cv2.resize(display, (640, 400))

                # Tkinter用に変換
                image = Image.fromarray(display)
                photo = ImageTk.PhotoImage(image)

                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.preview_canvas.image = photo

        # 次の更新をスケジュール
        self.root.after(33, self._update_preview)

    def _start_calibration(self, led_index):
        """光源キャリブレーション開始"""
        if not self.camera_opened:
            messagebox.showerror("エラー", "カメラが接続されていません")
            return

        # 現在のフレームを取得
        frame = self.camera.capture_frame()
        if frame is None:
            messagebox.showerror("エラー", "画像の取得に失敗しました")
            return

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # コールバック
        def on_calibration_complete(azimuth, elevation):
            self.light_vars[led_index]['azimuth'].set(round(azimuth, 1))
            self.light_vars[led_index]['elevation'].set(round(elevation, 1))
            self.status_var.set(f"LED{led_index + 1} キャリブレーション完了: 方位角={azimuth:.1f}°, 仰角={elevation:.1f}°")

        # キャリブレーションウィンドウを開く
        LightCalibrationWindow(self.root, gray, led_index, on_calibration_complete)

    def _capture_image(self, led_index):
        """画像撮影"""
        if not self.camera_opened:
            messagebox.showerror("エラー", "カメラが接続されていません")
            return

        frame = self.camera.capture_frame()
        if frame is not None:
            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.captured_images[led_index] = gray.astype(np.float32)

            # 表示更新
            self._update_captured_display(led_index)
            self.status_var.set(f"LED{led_index + 1} 撮影完了")
            logger.info(f"LED{led_index + 1} 撮影完了")
        else:
            messagebox.showerror("エラー", "撮影に失敗しました")

    def _update_captured_display(self, index):
        """撮影画像表示更新"""
        if self.captured_images[index] is not None:
            img = self.captured_images[index]
            # 正規化して表示
            display = (img / img.max() * 255).astype(np.uint8)
            display = cv2.resize(display, (200, 125))
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

            image = Image.fromarray(display)
            photo = ImageTk.PhotoImage(image)

            self.captured_canvases[index].delete("all")
            self.captured_canvases[index].create_image(0, 0, anchor=tk.NW, image=photo)
            self.captured_canvases[index].image = photo

    def _clear_all(self):
        """全画像クリア"""
        self.captured_images = [None, None, None]
        for canvas in self.captured_canvases:
            canvas.delete("all")
        self.normal_map = None
        self.albedo_map = None
        self.normal_canvas.delete("all")
        self.albedo_canvas.delete("all")
        self.status_var.set("クリアしました")

    def _get_light_vectors(self):
        """
        光源方向ベクトルを取得

        方位角(azimuth): XY平面上の角度（0度=正面、反時計回り）
        仰角(elevation): Z軸からの角度（90度=真上）

        Returns:
            3x3 numpy array: 各行が光源方向ベクトル [lx, ly, lz]
        """
        light_vectors = []
        for i in range(3):
            az = np.radians(self.light_vars[i]['azimuth'].get())
            el = np.radians(self.light_vars[i]['elevation'].get())

            # 球面座標から直交座標へ変換
            # lx = cos(elevation) * cos(azimuth)
            # ly = cos(elevation) * sin(azimuth)
            # lz = sin(elevation)
            lx = np.cos(el) * np.cos(az)
            ly = np.cos(el) * np.sin(az)
            lz = np.sin(el)

            light_vectors.append([lx, ly, lz])

        return np.array(light_vectors)

    def _calculate_normal_map(self):
        """
        法線マップを計算（Woodhamの方法）

        I = L * N * rho

        I: 輝度値ベクトル (3x1)
        L: 光源方向行列 (3x3)
        N: 法線ベクトル (3x1)
        rho: アルベド（反射率）

        N * rho = L^(-1) * I
        """
        # 全画像が撮影されているか確認
        if any(img is None for img in self.captured_images):
            missing = [i+1 for i, img in enumerate(self.captured_images) if img is None]
            messagebox.showerror("エラー", f"LED{missing} の画像が未撮影です")
            return

        self.status_var.set("法線マップ計算中...")
        self.root.update()

        try:
            # 光源方向行列
            L = self._get_light_vectors()
            logger.info(f"光源方向行列:\n{L}")

            # 逆行列を計算
            L_inv = np.linalg.inv(L)

            # 画像サイズ
            h, w = self.captured_images[0].shape

            # 輝度値を正規化（0-1）
            images = []
            for img in self.captured_images:
                # 小さい値をクリップ（ノイズ対策）
                img_norm = img / 255.0
                img_norm = np.clip(img_norm, 0.01, 1.0)
                images.append(img_norm)

            # 各画素の輝度値スタック (h, w, 3)
            I = np.stack(images, axis=-1)

            # 法線ベクトル計算 (h, w, 3)
            # N_rho = L^(-1) * I
            # reshape for matrix multiplication
            I_flat = I.reshape(-1, 3).T  # (3, h*w)
            N_rho_flat = L_inv @ I_flat  # (3, h*w)
            N_rho = N_rho_flat.T.reshape(h, w, 3)

            # アルベド（法線の大きさ）
            albedo = np.linalg.norm(N_rho, axis=2)
            albedo = np.clip(albedo, 0.001, None)  # ゼロ除算防止

            # 法線の正規化
            normal = N_rho / albedo[:, :, np.newaxis]

            # 法線マップを[-1,1]から[0,255]に変換して可視化
            # (nx, ny, nz) -> (R, G, B)
            # R = (nx + 1) / 2 * 255
            # G = (ny + 1) / 2 * 255
            # B = (nz + 1) / 2 * 255
            normal_vis = ((normal + 1) / 2 * 255).astype(np.uint8)

            # アルベドを正規化して可視化
            albedo_vis = (albedo / albedo.max() * 255).astype(np.uint8)

            self.normal_map = normal_vis
            self.albedo_map = albedo_vis

            # 結果表示
            self._display_result(normal_vis, self.normal_canvas)
            self._display_result(cv2.cvtColor(albedo_vis, cv2.COLOR_GRAY2RGB), self.albedo_canvas)

            self.status_var.set("法線マップ計算完了")
            logger.info("法線マップ計算完了")

        except np.linalg.LinAlgError as e:
            messagebox.showerror("エラー", f"行列計算エラー: {e}\n光源方向の設定を確認してください")
            self.status_var.set("計算エラー")
        except Exception as e:
            messagebox.showerror("エラー", f"計算エラー: {e}")
            self.status_var.set("計算エラー")
            logger.error(f"計算エラー: {e}")

    def _display_result(self, image, canvas):
        """結果画像をキャンバスに表示"""
        display = cv2.resize(image, (180, 112))
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

        img = Image.fromarray(display)
        photo = ImageTk.PhotoImage(img)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

    def _save_results(self):
        """結果を保存"""
        if self.normal_map is None:
            messagebox.showerror("エラー", "法線マップが計算されていません")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 法線マップ保存
        normal_path = os.path.join(self.output_dir, f"normal_map_{timestamp}.png")
        cv2.imwrite(normal_path, cv2.cvtColor(self.normal_map, cv2.COLOR_RGB2BGR))

        # アルベド保存
        albedo_path = os.path.join(self.output_dir, f"albedo_{timestamp}.png")
        cv2.imwrite(albedo_path, self.albedo_map)

        # 元画像も保存
        for i, img in enumerate(self.captured_images):
            if img is not None:
                img_path = os.path.join(self.output_dir, f"led{i+1}_{timestamp}.png")
                cv2.imwrite(img_path, img.astype(np.uint8))

        # 光源設定を保存
        config_path = os.path.join(self.output_dir, f"config_{timestamp}.txt")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("光源設定:\n")
            for i in range(3):
                az = self.light_vars[i]['azimuth'].get()
                el = self.light_vars[i]['elevation'].get()
                f.write(f"LED{i+1}: 方位角={az}°, 仰角={el}°\n")
            f.write(f"\n露出: {self.exposure_var.get()} us\n")
            f.write(f"ゲイン: {self.gain_var.get():.1f} dB\n")
            f.write(f"回転: {self.rotation}°\n")

        self.status_var.set(f"保存完了: {self.output_dir}")
        messagebox.showinfo("保存完了", f"保存先: {self.output_dir}")

    def cleanup(self):
        """終了処理"""
        if self.camera_opened:
            self.camera.close()
        logger.info("アプリケーション終了")


def main():
    root = tk.Tk()
    app = PhotometricStereoApp(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
