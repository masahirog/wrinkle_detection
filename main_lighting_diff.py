# -*- coding: utf-8 -*-
"""
照明差分法によるシワ検査システム
同軸照明と上方照明の差分からシワを検出
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import time
import os

from camera_control import StCameraControl
from wrinkle_detector_lighting import LightingDifferenceWrinkleDetector
from config_lighting import (CAMERA_SETTINGS, LIGHTING_DETECTION_PARAMS,
                             GUI_SETTINGS_LIGHTING, SAVE_SETTINGS_LIGHTING)
from utils import resize_for_display, ensure_directories


class LightingDifferenceApp:
    """照明差分法によるシワ検査アプリケーション"""

    def __init__(self, root):
        """初期化"""
        self.root = root
        self.root.title(GUI_SETTINGS_LIGHTING['window_title'])

        # カメラコントロール
        self.camera = StCameraControl()

        # シワ検出器
        self.detector = LightingDifferenceWrinkleDetector(
            threshold=LIGHTING_DETECTION_PARAMS['difference_threshold'],
            min_length=LIGHTING_DETECTION_PARAMS['min_wrinkle_length']
        )

        # 撮影画像
        self.img_coaxial = None
        self.img_top = None

        # 検出結果
        self.wrinkle_mask = None
        self.diff_image = None
        self.debug_images = None
        self.stats = None

        # 状態フラグ
        self.is_running = False

        # ディレクトリの初期化
        self._ensure_directories()

        # GUI構築
        self.build_gui()

        # ウィンドウ閉じるボタンのハンドラを設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # カメラを自動スキャン
        self.scan_cameras()

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(SAVE_SETTINGS_LIGHTING['results_dir'], exist_ok=True)
        os.makedirs(SAVE_SETTINGS_LIGHTING['debug_dir'], exist_ok=True)

    def build_gui(self):
        """GUIを構築"""

        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左側: プレビューエリア
        preview_frame = ttk.LabelFrame(main_frame, text="カメラプレビュー", padding="10")
        preview_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky=(tk.N, tk.S))

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack()

        # 右上: カメラ・照明制御
        control_frame = ttk.LabelFrame(main_frame, text="カメラ・照明制御", padding="10")
        control_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        # カメラ選択
        ttk.Label(control_frame, text="カメラ:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            state='readonly',
            width=30
        )
        self.camera_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # カメラ再検出ボタン
        self.rescan_button = ttk.Button(control_frame, text="再検出", command=self.scan_cameras)
        self.rescan_button.grid(row=0, column=3, padx=5, pady=5)

        # カメラ起動/停止ボタン
        self.start_button = ttk.Button(control_frame, text="カメラ起動", command=self.start_camera)
        self.start_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.stop_button = ttk.Button(control_frame, text="カメラ停止", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 撮影制御
        capture_frame = ttk.LabelFrame(control_frame, text="撮影制御", padding="5")
        capture_frame.grid(row=2, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))

        self.capture_coaxial_btn = ttk.Button(capture_frame, text="同軸照明で撮影",
                                              command=self.capture_coaxial, state=tk.DISABLED)
        self.capture_coaxial_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.capture_top_btn = ttk.Button(capture_frame, text="上方照明で撮影",
                                          command=self.capture_top, state=tk.DISABLED)
        self.capture_top_btn.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 撮影状況表示
        self.capture_status_label = ttk.Label(capture_frame, text="同軸: 未撮影 | 上方: 未撮影",
                                             font=('Arial', 9))
        self.capture_status_label.grid(row=1, column=0, columnspan=2, pady=5)

        # 右下: シワ検出
        detection_frame = ttk.LabelFrame(main_frame, text="シワ検出", padding="10")
        detection_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))

        # 検出方法選択
        self.detection_method_var = tk.StringVar(value="advanced")
        ttk.Radiobutton(detection_frame, text="基本検出", variable=self.detection_method_var,
                       value="basic").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(detection_frame, text="高度な検出（横シワ強調）", variable=self.detection_method_var,
                       value="advanced").grid(row=0, column=1, sticky=tk.W)

        # シワ検出実行ボタン
        self.detect_button = ttk.Button(detection_frame, text="シワ検出実行",
                                        command=self.detect_wrinkles, state=tk.DISABLED)
        self.detect_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky=(tk.W, tk.E))

        # 結果表示ボタン
        self.show_result_button = ttk.Button(detection_frame, text="結果詳細表示",
                                            command=self.show_result_window, state=tk.DISABLED)
        self.show_result_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 検出結果表示
        self.result_label = ttk.Label(detection_frame, text="", font=('Arial', 10),
                                     foreground="black")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

    def scan_cameras(self):
        """カメラをスキャン"""
        if self.is_running:
            messagebox.showwarning("警告", "カメラを停止してから再検出してください")
            return

        self.available_cameras = self.camera.scan_available_cameras()
        camera_names = [cam['name'] for cam in self.available_cameras]
        self.camera_combo['values'] = camera_names

        if camera_names:
            self.camera_combo.current(0)
        else:
            messagebox.showwarning("警告", "利用可能なカメラが見つかりませんでした")

    def start_camera(self):
        """カメラを起動"""
        selected_index = self.camera_combo.current()
        if selected_index < 0 or not self.available_cameras:
            messagebox.showerror("エラー", "カメラを選択してください")
            return

        camera_info = self.available_cameras[selected_index]
        camera_index = camera_info['index']

        if self.camera.open(camera_index):
            self.is_running = True

            # カメラ設定を適用
            self.camera.set_exposure(CAMERA_SETTINGS['exposure_time'])
            self.camera.set_gain(CAMERA_SETTINGS['gain'])
            self.camera.set_brightness(CAMERA_SETTINGS['brightness'])

            # ボタン状態変更
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.capture_coaxial_btn.config(state=tk.NORMAL)
            self.capture_top_btn.config(state=tk.NORMAL)

            # プレビュー更新を開始
            self.update_preview()
        else:
            messagebox.showerror("エラー", "カメラの起動に失敗しました")

    def stop_camera(self):
        """カメラを停止"""
        self.is_running = False
        time.sleep(0.1)
        self.camera.close()

        # ボタン状態変更
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.capture_coaxial_btn.config(state=tk.DISABLED)
        self.capture_top_btn.config(state=tk.DISABLED)

    def update_preview(self):
        """プレビューを更新"""
        if not self.is_running:
            return

        frame = self.camera.capture_frame()

        if frame is not None:
            # リサイズして表示
            display_frame = resize_for_display(
                frame,
                GUI_SETTINGS_LIGHTING['preview_width'],
                GUI_SETTINGS_LIGHTING['preview_height']
            )

            # BGR → RGB変換
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # PIL Image → ImageTk
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)

            # ラベルに表示
            self.preview_label.config(image=photo)
            self.preview_label.image = photo

        # 30fps程度で更新
        self.root.after(33, self.update_preview)

    def capture_coaxial(self):
        """同軸照明で撮影"""
        self.img_coaxial = self.camera.capture_frame()

        if self.img_coaxial is not None:
            self.update_capture_status()
            self.check_detection_ready()
            messagebox.showinfo("撮影完了", "同軸照明で撮影しました")

    def capture_top(self):
        """上方照明で撮影"""
        self.img_top = self.camera.capture_frame()

        if self.img_top is not None:
            self.update_capture_status()
            self.check_detection_ready()
            messagebox.showinfo("撮影完了", "上方照明で撮影しました")

    def update_capture_status(self):
        """撮影状況を更新"""
        coaxial_status = "撮影済" if self.img_coaxial is not None else "未撮影"
        top_status = "撮影済" if self.img_top is not None else "未撮影"
        self.capture_status_label.config(text=f"同軸: {coaxial_status} | 上方: {top_status}")

    def check_detection_ready(self):
        """シワ検出が実行可能かチェック"""
        if self.img_coaxial is not None and self.img_top is not None:
            self.detect_button.config(state=tk.NORMAL)

    def detect_wrinkles(self):
        """シワ検出を実行"""
        if self.img_coaxial is None or self.img_top is None:
            messagebox.showerror("エラー", "両方の画像を撮影してください")
            return

        method = self.detection_method_var.get()

        try:
            if method == "basic":
                # 基本検出
                self.wrinkle_mask, self.diff_image, self.debug_images = \
                    self.detector.detect_wrinkles(self.img_coaxial, self.img_top)
            else:
                # 高度な検出
                self.wrinkle_mask, self.diff_image, self.debug_images = \
                    self.detector.detect_with_horizontal_emphasis(self.img_coaxial, self.img_top)

            # 統計情報を計算
            self.stats = self.detector.analyze_wrinkles(self.wrinkle_mask)

            # 結果を表示
            self.display_result()

            # 結果詳細表示ボタンを有効化
            self.show_result_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("エラー", f"シワ検出に失敗しました: {e}")

    def display_result(self):
        """検出結果を表示"""
        if self.stats is None:
            return

        # NG判定
        is_ng = (self.stats['wrinkle_ratio'] >= LIGHTING_DETECTION_PARAMS['ng_wrinkle_ratio'] or
                self.stats['wrinkle_count'] >= LIGHTING_DETECTION_PARAMS['ng_wrinkle_count'])

        result_text = "NG" if is_ng else "OK"
        result_color = "red" if is_ng else "green"

        result_info = f"""結果: {result_text}
シワ数: {self.stats['wrinkle_count']}
シワ面積率: {self.stats['wrinkle_ratio']:.2f}%
平均長さ: {self.stats['avg_length']:.1f}px"""

        self.result_label.config(text=result_info, foreground=result_color)

    def show_result_window(self):
        """結果詳細ウィンドウを表示"""
        if self.debug_images is None:
            messagebox.showwarning("警告", "検出結果がありません")
            return

        # 新しいウィンドウを作成
        result_window = tk.Toplevel(self.root)
        result_window.title("シワ検出結果 - 詳細")
        result_window.geometry("900x700")

        # スクロール可能なキャンバス
        canvas = tk.Canvas(result_window, bg='white')
        v_scrollbar = ttk.Scrollbar(result_window, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=v_scrollbar.set)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)

        # デバッグ画像を表示
        for idx, (key, img) in enumerate(self.debug_images.items()):
            row = idx // 2
            col = idx % 2

            frame = ttk.LabelFrame(scrollable_frame, text=key, padding="5")
            frame.grid(row=row, column=col, padx=5, pady=5)

            # グレースケールの場合はカラーに変換
            if len(img.shape) == 2:
                img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # リサイズ
            display_width = 400
            display_height = int(img_display.shape[0] * display_width / img_display.shape[1])
            img_resized = cv2.resize(img_display, (display_width, display_height))

            pil_image = Image.fromarray(img_resized)
            photo = ImageTk.PhotoImage(image=pil_image)

            label = ttk.Label(frame, image=photo)
            label.image = photo
            label.pack()

        # 閉じるボタン
        ttk.Button(scrollable_frame, text="閉じる", command=result_window.destroy).grid(
            row=(len(self.debug_images) + 1) // 2, column=0, columnspan=2, pady=10
        )

        scrollable_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def on_closing(self):
        """ウィンドウを閉じる時の処理"""
        if self.is_running:
            self.is_running = False
            time.sleep(0.1)
            self.camera.close()

        self.root.destroy()


def main():
    """メイン関数"""
    root = tk.Tk()
    app = LightingDifferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
