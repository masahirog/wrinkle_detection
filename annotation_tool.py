# -*- coding: utf-8 -*-
"""
シワアノテーションツール
画像上でシワ領域をポリゴンで囲み、YOLO形式で保存
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os


class WrinkleAnnotationTool:
    """シワアノテーションツール"""

    def __init__(self, parent, image_path, save_callback):
        """
        初期化

        Args:
            parent: 親ウィンドウ
            image_path: アノテーション対象の画像パス
            save_callback: 保存時のコールバック関数
        """
        self.window = tk.Toplevel(parent)
        self.window.title(f"シワアノテーション - {os.path.basename(image_path)}")
        self.window.geometry("1200x800")

        self.image_path = image_path
        self.save_callback = save_callback

        # 画像を読み込み
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            messagebox.showerror("エラー", f"画像の読み込みに失敗しました: {image_path}")
            self.window.destroy()
            return

        self.display_image = self.original_image.copy()
        self.image_height, self.image_width = self.original_image.shape[:2]

        # アノテーションデータ
        self.polygons = []  # 完成したポリゴンのリスト
        self.current_polygon = []  # 現在描画中のポリゴン

        # 表示倍率
        self.scale = 1.0

        # GUI構築
        self.build_gui()

        # キーバインド
        self.window.bind('<Escape>', lambda e: self.cancel_current_polygon())
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def build_gui(self):
        """GUIを構築"""
        # 左側: キャンバス
        canvas_frame = ttk.Frame(self.window)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # スクロールバー付きキャンバス
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(canvas_frame, bg='gray',
                               xscrollcommand=h_scroll.set,
                               yscrollcommand=v_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

        # キャンバスにクリックイベントをバインド
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.complete_polygon)  # 右クリックで完了

        # 画像を表示
        self.update_canvas()

        # 右側: コントロールパネル
        control_frame = ttk.Frame(self.window, padding="10")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(control_frame, text="アノテーション操作", font=('Arial', 12, 'bold')).pack(pady=10)

        ttk.Label(control_frame, text="1. 左クリック: ポリゴンの頂点を追加\n"
                                     "2. 右クリック: ポリゴンを完成\n"
                                     "3. ESC: 現在のポリゴンをキャンセル",
                 justify=tk.LEFT).pack(pady=10, anchor=tk.W)

        # ポリゴン数表示
        self.polygon_count_label = ttk.Label(control_frame, text="アノテーション済み: 0個",
                                            font=('Arial', 10))
        self.polygon_count_label.pack(pady=10)

        # ボタン
        ttk.Button(control_frame, text="最後のポリゴンを削除",
                  command=self.delete_last_polygon).pack(pady=5, fill=tk.X)

        ttk.Button(control_frame, text="全てクリア",
                  command=self.clear_all).pack(pady=5, fill=tk.X)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="保存して閉じる",
                  command=self.save_and_close).pack(pady=5, fill=tk.X)

        ttk.Button(control_frame, text="保存せずに閉じる",
                  command=self.window.destroy).pack(pady=5, fill=tk.X)

    def update_canvas(self):
        """キャンバスを更新"""
        # 画像を描画
        img_display = self.display_image.copy()

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        # PIL Image → ImageTk
        pil_image = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(image=pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # スクロール範囲を設定
        self.canvas.config(scrollregion=(0, 0, self.image_width, self.image_height))

    def on_canvas_click(self, event):
        """キャンバスクリック時"""
        # スクロール位置を考慮
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # 頂点を追加
        self.current_polygon.append((int(x), int(y)))

        # 描画更新
        self.redraw_annotations()

    def complete_polygon(self, event=None):
        """現在のポリゴンを完成"""
        if len(self.current_polygon) < 3:
            messagebox.showwarning("警告", "ポリゴンは3点以上必要です")
            return

        # 完成したポリゴンを保存
        self.polygons.append(self.current_polygon.copy())
        self.current_polygon = []

        # 描画更新
        self.redraw_annotations()

        # ポリゴン数を更新
        self.polygon_count_label.config(text=f"アノテーション済み: {len(self.polygons)}個")

    def cancel_current_polygon(self):
        """現在のポリゴンをキャンセル"""
        self.current_polygon = []
        self.redraw_annotations()

    def delete_last_polygon(self):
        """最後のポリゴンを削除"""
        if self.polygons:
            self.polygons.pop()
            self.redraw_annotations()
            self.polygon_count_label.config(text=f"アノテーション済み: {len(self.polygons)}個")

    def clear_all(self):
        """全てのアノテーションをクリア"""
        result = messagebox.askyesno("確認", "全てのアノテーションを削除しますか？")
        if result:
            self.polygons = []
            self.current_polygon = []
            self.redraw_annotations()
            self.polygon_count_label.config(text=f"アノテーション済み: 0個")

    def redraw_annotations(self):
        """アノテーションを再描画"""
        self.display_image = self.original_image.copy()

        # 完成したポリゴンを描画（緑色）
        for polygon in self.polygons:
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(self.display_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(self.display_image, [pts], color=(0, 255, 0, 50))  # 半透明

        # 現在のポリゴンを描画（黄色）
        if len(self.current_polygon) > 0:
            for i, pt in enumerate(self.current_polygon):
                cv2.circle(self.display_image, pt, 5, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(self.display_image, self.current_polygon[i-1], pt, (0, 255, 255), 2)

        self.update_canvas()

    def save_and_close(self):
        """保存して閉じる"""
        if len(self.polygons) == 0:
            result = messagebox.askyesno("確認", "アノテーションがありません。OK品として保存しますか？")
            if not result:
                return

        # YOLO形式で保存
        self.save_yolo_format()

        # コールバックを呼ぶ
        if self.save_callback:
            self.save_callback(self.image_path, self.polygons)

        self.window.destroy()

    def save_yolo_format(self):
        """YOLO形式でアノテーションを保存"""
        # 画像ファイル名から拡張子を除いたベース名を取得
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # ラベルファイルのパス（同じディレクトリに.txt）
        label_dir = os.path.dirname(self.image_path)
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        # YOLO形式で保存（正規化座標）
        with open(label_path, 'w') as f:
            for polygon in self.polygons:
                # クラスID（0 = wrinkle）
                class_id = 0

                # ポリゴンの座標を正規化（0-1）
                normalized_coords = []
                for x, y in polygon:
                    norm_x = x / self.image_width
                    norm_y = y / self.image_height
                    normalized_coords.append(f"{norm_x:.6f}")
                    normalized_coords.append(f"{norm_y:.6f}")

                # YOLO形式: class_id x1 y1 x2 y2 x3 y3 ...
                line = f"{class_id} " + " ".join(normalized_coords) + "\n"
                f.write(line)

        print(f"アノテーション保存: {label_path}")

    def on_closing(self):
        """ウィンドウを閉じる時"""
        if len(self.polygons) > 0 or len(self.current_polygon) > 0:
            result = messagebox.askyesnocancel("確認", "アノテーションを保存しますか？")
            if result is True:
                self.save_and_close()
            elif result is False:
                self.window.destroy()
            # Cancelの場合は何もしない
        else:
            self.window.destroy()
