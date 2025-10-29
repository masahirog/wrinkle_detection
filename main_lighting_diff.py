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
            min_length=LIGHTING_DETECTION_PARAMS['min_wrinkle_length'],
            sobel_threshold=LIGHTING_DETECTION_PARAMS['sobel_threshold']
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
        self.detection_done = False  # 検出が完了しているか

        # パラメータ変数
        self.param_threshold = tk.IntVar(value=LIGHTING_DETECTION_PARAMS['difference_threshold'])
        self.param_min_length = tk.IntVar(value=LIGHTING_DETECTION_PARAMS['min_wrinkle_length'])
        self.param_sobel_threshold = tk.IntVar(value=LIGHTING_DETECTION_PARAMS['sobel_threshold'])

        # ディレクトリの初期化
        self._ensure_directories()

        # GUI構築
        self.build_gui()

        # ウィンドウ閉じるボタンのハンドラを設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # YOLOセグメンテーションモデルを事前ロード
        print("YOLOセグメンテーションモデルを事前ロード中...")
        from wrinkle_detector_lighting import get_yolo_seg_model
        get_yolo_seg_model()
        print("YOLOセグメンテーションモデルのロード完了")

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
                       value="basic", command=self.on_method_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(detection_frame, text="高度な検出（横シワ強調）", variable=self.detection_method_var,
                       value="advanced", command=self.on_method_change).grid(row=0, column=1, sticky=tk.W)

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

        # パラメータ調整フレーム（メインフレームの下段）
        param_frame = ttk.LabelFrame(main_frame, text="パラメータ調整（リアルタイム反映）", padding="10")
        param_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 差分閾値スライダー
        ttk.Label(param_frame, text="差分閾値:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.threshold_scale = ttk.Scale(param_frame, from_=5, to=100, variable=self.param_threshold,
                                         orient=tk.HORIZONTAL, length=200,
                                         command=self.on_param_change)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=5)
        self.threshold_label = ttk.Label(param_frame, text=f"{self.param_threshold.get()}")
        self.threshold_label.grid(row=0, column=2, pady=5)
        ttk.Label(param_frame, text="大きいほど強いシワのみ検出", font=('Arial', 8), foreground="gray").grid(
            row=0, column=3, sticky=tk.W, padx=10)

        # 最小シワ長さスライダー
        ttk.Label(param_frame, text="最小シワ長さ:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.min_length_scale = ttk.Scale(param_frame, from_=5, to=50, variable=self.param_min_length,
                                          orient=tk.HORIZONTAL, length=200,
                                          command=self.on_param_change)
        self.min_length_scale.grid(row=1, column=1, padx=5, pady=5)
        self.min_length_label = ttk.Label(param_frame, text=f"{self.param_min_length.get()}px")
        self.min_length_label.grid(row=1, column=2, pady=5)
        ttk.Label(param_frame, text="大きいほど短いシワを無視", font=('Arial', 8), foreground="gray").grid(
            row=1, column=3, sticky=tk.W, padx=10)

        # Sobel閾値スライダー
        ttk.Label(param_frame, text="Sobel閾値:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.sobel_scale = ttk.Scale(param_frame, from_=5, to=50, variable=self.param_sobel_threshold,
                                     orient=tk.HORIZONTAL, length=200,
                                     command=self.on_param_change)
        self.sobel_scale.grid(row=2, column=1, padx=5, pady=5)
        self.sobel_label = ttk.Label(param_frame, text=f"{self.param_sobel_threshold.get()}")
        self.sobel_label.grid(row=2, column=2, pady=5)
        ttk.Label(param_frame, text="横線検出の感度（上方照明）", font=('Arial', 8), foreground="gray").grid(
            row=2, column=3, sticky=tk.W, padx=10)

        # 処理解説フレーム
        explain_frame = ttk.LabelFrame(main_frame, text="処理の解説", padding="10")
        explain_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        # スクロール可能なテキストエリア
        text_scroll = ttk.Scrollbar(explain_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.explain_text = tk.Text(explain_frame, height=8, width=100, wrap=tk.WORD,
                                   yscrollcommand=text_scroll.set, font=('Arial', 9))
        self.explain_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.explain_text.yview)

        # 初期説明文を表示
        self.update_explanation()

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

    def on_param_change(self, value):
        """パラメータ変更時のコールバック"""
        # ラベルを更新
        self.threshold_label.config(text=f"{self.param_threshold.get()}")
        self.min_length_label.config(text=f"{self.param_min_length.get()}px")
        self.sobel_label.config(text=f"{self.param_sobel_threshold.get()}")

        # 検出が既に完了している場合は自動的に再計算
        if self.detection_done:
            self.detect_wrinkles()

    def on_method_change(self):
        """検出方法変更時のコールバック"""
        # 検出が既に完了している場合は自動的に再計算
        if self.detection_done:
            self.detect_wrinkles()

    def update_explanation(self):
        """処理解説を更新"""
        explanation = """【照明差分法によるシワ検出の仕組み - 改良版】

■ 基本原理（柄とシワの区別方法）
・上方照明で暗い線が検出される → シワ候補 または 柄の黒い模様
・同軸照明との差分で判定：
  - 同軸照明でも暗い → 柄の色（黒い模様）→ 除外
  - 同軸照明では明るい → シワの影 → シワと判定

■ 処理ステップ

【基本検出】
1. YOLOセグメンテーション
   → ボトルの正確な形状を検出し、背景を除外

2. 上方照明画像で線状の暗い部分を抽出
   → エッジ検出 + 横方向の連続性チェック
   → 現在の値: 最小長さ {min_length}px
   → これが「シワ候補 + 柄」

3. 同軸照明との差分を計算
   → 各線の位置で明るさを比較

4. 差分が大きい線のみを残す
   → 現在の値: 差分閾値 {threshold}
   → 差分が大きい = シワの影
   → 差分が小さい = 柄の色 → 除外

5. ボトルマスク適用
   → ボトル領域内のシワのみを残す

【高度な検出】
1-3. 同上

4. Sobelフィルタで上方照明の横エッジを検出
   → 現在の値: Sobel閾値 {sobel}
   → 上方照明画像に対してSobelを適用
   → 縦方向の明るさ変化 = 横線

5. 横方向連続性チェック
   → 最小長さ {min_length}px 以上の横線のみ

6. 差分が大きい線のみを残す
   → 差分閾値 {threshold} 以上 → シワ
   → 差分閾値未満 → 柄 → 除外

7. ボトルマスク適用
   → ボトル領域内のシワのみ

■ 最終判定
→ NG判定：シワ数が{ng_count}本以上 または シワ率が{ng_ratio}%以上
""".format(
            threshold=self.param_threshold.get(),
            sobel=self.param_sobel_threshold.get(),
            min_length=self.param_min_length.get(),
            ng_count=LIGHTING_DETECTION_PARAMS['ng_wrinkle_count'],
            ng_ratio=LIGHTING_DETECTION_PARAMS['ng_wrinkle_ratio']
        )

        self.explain_text.delete(1.0, tk.END)
        self.explain_text.insert(1.0, explanation)

    def detect_wrinkles(self):
        """シワ検出を実行"""
        if self.img_coaxial is None or self.img_top is None:
            if not self.detection_done:  # 初回のみエラー表示
                messagebox.showerror("エラー", "両方の画像を撮影してください")
            return

        # 現在のパラメータで検出器を更新
        self.detector.threshold = self.param_threshold.get()
        self.detector.min_length = self.param_min_length.get()
        self.detector.sobel_threshold = self.param_sobel_threshold.get()

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

            # 検出完了フラグを設定
            self.detection_done = True

            # 説明文を更新
            self.update_explanation()

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
