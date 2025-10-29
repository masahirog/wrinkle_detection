# -*- coding: utf-8 -*-
"""
照明差分法によるシワ検出
同軸照明と上方照明の差分からシワを検出
"""

import cv2
import numpy as np
import logging

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLOセグメンテーションモデルのグローバル変数（初回のみロード）
_yolo_seg_model = None


def get_yolo_seg_model():
    """
    YOLOセグメンテーションモデルを取得（シングルトンパターン）

    Returns:
        YOLOセグメンテーションモデル
    """
    global _yolo_seg_model
    if _yolo_seg_model is None:
        try:
            from ultralytics import YOLO
            logger.info("YOLOセグメンテーションモデルをロード中...")
            _yolo_seg_model = YOLO('yolov8n-seg.pt')  # セグメンテーションモデル
            logger.info("YOLOセグメンテーションモデルのロード完了")
        except Exception as e:
            logger.error(f"YOLOセグメンテーションモデルのロードに失敗: {e}")
            _yolo_seg_model = None
    return _yolo_seg_model


def detect_bottle_segmentation(image):
    """
    YOLOセグメンテーションでボトルを検出し、マスクを取得

    Args:
        image: 入力画像

    Returns:
        bottle_mask: ボトルのマスク画像（白=ボトル、黒=背景）
        annotated_image: セグメンテーション結果を描画した画像
    """
    model = get_yolo_seg_model()

    if model is None:
        # モデルがない場合は全体を対象にする
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255, image.copy()

    try:
        # YOLO推論（verbose=Falseでログを抑制）
        results = model(image, verbose=False)

        h, w = image.shape[:2]
        bottle_mask = np.zeros((h, w), dtype=np.uint8)
        annotated_image = image.copy()

        # 検出結果を処理
        for result in results:
            if result.masks is None:
                continue

            boxes = result.boxes
            masks = result.masks

            for i, box in enumerate(boxes):
                # クラスID取得（39 = bottle）
                cls = int(box.cls[0])
                if cls == 39:  # bottle
                    # マスクデータを取得
                    mask_data = masks.data[i].cpu().numpy()

                    # マスクを元の画像サイズにリサイズ
                    mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)

                    # マスクを統合（複数のボトルがある場合はOR結合）
                    bottle_mask = cv2.bitwise_or(bottle_mask, (mask_resized * 255).astype(np.uint8))

                    # 信頼度
                    confidence = float(box.conf[0])

                    # セグメンテーション結果を描画（緑色の半透明オーバーレイ）
                    mask_colored = np.zeros_like(annotated_image)
                    mask_colored[:, :, 1] = (mask_resized * 255).astype(np.uint8)  # 緑チャンネル
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, mask_colored, 0.3, 0)

                    # バウンディングボックスも描画
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 信頼度を表示
                    label = f"Bottle {confidence:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return bottle_mask, annotated_image

    except Exception as e:
        logger.error(f"YOLOセグメンテーションエラー: {e}")
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255, image.copy()


class LightingDifferenceWrinkleDetector:
    """
    照明差分法によるシワ検出器

    同軸照明と上方照明の差分から、横シワの影を検出します
    """

    def __init__(self, threshold=20, min_length=20, sobel_threshold=10):
        """
        初期化

        Args:
            threshold: 差分の閾値（シワと柄を区別する閾値）
            min_length: 検出する最小シワ長さ（ピクセル）
            sobel_threshold: Sobelフィルタの閾値（横線検出の感度）
        """
        self.threshold = threshold
        self.min_length = min_length
        self.sobel_threshold = sobel_threshold

        # 画像位置合わせ用（照明変化に強い）
        try:
            self.matcher = cv2.createAlignMTB()
        except:
            self.matcher = None
            logger.warning("AlignMTBが利用できません。位置合わせなしで動作します")

    def align_images(self, img1, img2):
        """
        画像の位置合わせ

        Args:
            img1: 画像1
            img2: 画像2

        Returns:
            位置合わせされた画像のタプル
        """
        if self.matcher is None:
            return img1, img2

        try:
            aligned = self.matcher.process([img1, img2])
            return aligned[0], aligned[1]
        except Exception as e:
            logger.warning(f"位置合わせに失敗: {e}")
            return img1, img2

    def detect_wrinkles(self, img_coaxial, img_top):
        """
        基本的なシワ検出（新アプローチ）

        1. 上方照明画像で線状の暗い部分を抽出
        2. その線の位置で同軸照明との差分を計算
        3. 差分が大きい → シワ、差分が小さい → 柄

        Args:
            img_coaxial: 同軸照明画像
            img_top: 上方照明画像

        Returns:
            wrinkle_mask: シワマスク画像
            diff: 差分画像
            debug_images: デバッグ用画像の辞書
        """
        debug_images = {}

        # 元画像を保存
        debug_images['1_coaxial_original'] = img_coaxial.copy()
        debug_images['2_top_original'] = img_top.copy()

        # YOLOセグメンテーションでボトル領域を検出
        bottle_mask, bottle_seg_image = detect_bottle_segmentation(img_coaxial)
        debug_images['3_bottle_segmentation'] = bottle_seg_image

        # グレースケール変換
        gray_coax = cv2.cvtColor(img_coaxial, cv2.COLOR_BGR2GRAY)
        gray_top = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)

        debug_images['4_coaxial_gray'] = gray_coax
        debug_images['5_top_gray'] = gray_top

        # ステップ1: 上方照明画像で線状の暗い部分（シワ候補）を抽出
        # 2値化（暗い部分を抽出）
        _, dark_areas = cv2.threshold(gray_top, 127, 255, cv2.THRESH_BINARY_INV)
        debug_images['6_top_dark_areas'] = dark_areas

        # エッジ検出（横方向の線を強調）
        edges = cv2.Canny(gray_top, 50, 150)
        debug_images['7_top_edges'] = edges

        # 横長カーネルで横方向の線を強調
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (self.min_length, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
        debug_images['8_horizontal_lines'] = horizontal_lines

        # ステップ2: 各線の位置で同軸照明との差分を計算
        diff = cv2.absdiff(gray_coax, gray_top)
        debug_images['9_difference'] = diff

        # ステップ3: 差分が大きい部分のみを残す（シワと柄を区別）
        # 線の位置で差分が閾値以上 → シワ
        _, diff_binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        debug_images['10_diff_threshold'] = diff_binary

        # 線と差分の両方が検出された部分のみをシワとする
        wrinkle_mask = cv2.bitwise_and(horizontal_lines, diff_binary)
        debug_images['11_wrinkle_candidates'] = wrinkle_mask

        # ボトルマスクを適用（ボトル領域外を除外）
        wrinkle_mask = cv2.bitwise_and(wrinkle_mask, bottle_mask)
        debug_images['12_masked_wrinkles'] = wrinkle_mask

        return wrinkle_mask, diff, debug_images

    def detect_with_horizontal_emphasis(self, img_coaxial, img_top):
        """
        横シワを強調した高度な検出（新アプローチ）

        1. 上方照明画像で線状の暗い部分を抽出（Sobel + 横線強調）
        2. その線の位置で同軸照明との差分を計算
        3. 差分が大きい → シワ、差分が小さい → 柄

        Args:
            img_coaxial: 同軸照明画像
            img_top: 上方照明画像

        Returns:
            horizontal_features: 横シワ特徴画像
            diff_normalized: 差分画像
            debug_images: デバッグ用画像の辞書
        """
        debug_images = {}

        # 元画像を保存
        debug_images['1_coaxial_original'] = img_coaxial.copy()
        debug_images['2_top_original'] = img_top.copy()

        # YOLOセグメンテーションでボトル領域を検出
        bottle_mask, bottle_seg_image = detect_bottle_segmentation(img_coaxial)
        debug_images['3_bottle_segmentation'] = bottle_seg_image

        # グレースケール変換
        gray_coax = cv2.cvtColor(img_coaxial, cv2.COLOR_BGR2GRAY)
        gray_top = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)

        debug_images['4_coaxial_gray'] = gray_coax
        debug_images['5_top_gray'] = gray_top

        # ステップ1: 上方照明画像でSobelフィルタ（横エッジ抽出）
        sobel_y_top = cv2.Sobel(gray_top, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y_abs = np.abs(sobel_y_top)
        sobel_y_norm = cv2.normalize(sobel_y_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug_images['6_top_sobel'] = sobel_y_norm

        # Sobel結果を閾値処理
        _, sobel_binary = cv2.threshold(sobel_y_norm, self.sobel_threshold, 255, cv2.THRESH_BINARY)
        debug_images['7_sobel_threshold'] = sobel_binary

        # 横方向に連続性があるものを抽出（横線のみ）
        kernel_h = np.ones((1, self.min_length), np.uint8)
        horizontal_lines = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel_h)
        debug_images['8_horizontal_lines'] = horizontal_lines

        # ステップ2: 差分計算
        diff = cv2.absdiff(gray_coax, gray_top)
        debug_images['9_difference'] = diff

        # ステップ3: 差分が大きい部分のみを残す（シワと柄を区別）
        _, diff_binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        debug_images['10_diff_threshold'] = diff_binary

        # 横線と差分の両方が検出された部分のみをシワとする
        wrinkle_mask = cv2.bitwise_and(horizontal_lines, diff_binary)
        debug_images['11_wrinkle_candidates'] = wrinkle_mask

        # さらにノイズ除去
        kernel_clean = np.ones((3, 3), np.uint8)
        wrinkle_mask = cv2.morphologyEx(wrinkle_mask, cv2.MORPH_OPEN, kernel_clean)
        debug_images['12_cleaned_wrinkles'] = wrinkle_mask

        # ボトルマスクを適用（ボトル領域外を除外）
        wrinkle_mask = cv2.bitwise_and(wrinkle_mask, bottle_mask)
        debug_images['13_masked_wrinkles'] = wrinkle_mask

        return wrinkle_mask, diff, debug_images

    def analyze_wrinkles(self, wrinkle_mask):
        """
        シワの統計情報を分析

        Args:
            wrinkle_mask: シワマスク画像

        Returns:
            dict: 統計情報
        """
        # 輪郭検出
        contours, _ = cv2.findContours(wrinkle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # シワの数
        wrinkle_count = len(contours)

        # シワのピクセル数
        wrinkle_pixels = np.count_nonzero(wrinkle_mask)

        # 全体のピクセル数
        total_pixels = wrinkle_mask.shape[0] * wrinkle_mask.shape[1]

        # シワの面積率
        wrinkle_ratio = (wrinkle_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # 各シワの長さと面積
        wrinkle_lengths = []
        wrinkle_areas = []

        for contour in contours:
            # バウンディングボックス
            x, y, w, h = cv2.boundingRect(contour)
            wrinkle_lengths.append(w)
            wrinkle_areas.append(cv2.contourArea(contour))

        # 平均値
        avg_length = np.mean(wrinkle_lengths) if wrinkle_lengths else 0
        avg_area = np.mean(wrinkle_areas) if wrinkle_areas else 0

        stats = {
            'wrinkle_count': wrinkle_count,
            'wrinkle_pixels': wrinkle_pixels,
            'total_pixels': total_pixels,
            'wrinkle_ratio': wrinkle_ratio,
            'avg_length': avg_length,
            'avg_area': avg_area,
            'max_length': max(wrinkle_lengths) if wrinkle_lengths else 0,
            'max_area': max(wrinkle_areas) if wrinkle_areas else 0
        }

        return stats

    def draw_result(self, image, wrinkle_mask, stats):
        """
        結果を画像に描画

        Args:
            image: 元画像
            wrinkle_mask: シワマスク
            stats: 統計情報

        Returns:
            result_image: 結果画像
        """
        result_image = image.copy()

        # シワマスクを赤色でオーバーレイ
        mask_colored = np.zeros_like(result_image)
        mask_colored[:, :, 2] = wrinkle_mask  # 赤チャンネル

        # 半透明で合成
        result_image = cv2.addWeighted(result_image, 0.7, mask_colored, 0.3, 0)

        # 統計情報をテキスト表示
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        texts = [
            f"Wrinkle Count: {stats['wrinkle_count']}",
            f"Wrinkle Ratio: {stats['wrinkle_ratio']:.2f}%",
            f"Avg Length: {stats['avg_length']:.1f}px",
            f"Max Length: {stats['max_length']:.1f}px"
        ]

        for i, text in enumerate(texts):
            cv2.putText(result_image, text, (10, y_offset + i * 30),
                       font, 0.7, (0, 255, 0), 2)

        return result_image
