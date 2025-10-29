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


class LightingDifferenceWrinkleDetector:
    """
    照明差分法によるシワ検出器

    同軸照明と上方照明の差分から、横シワの影を検出します
    """

    def __init__(self, threshold=20, min_length=20):
        """
        初期化

        Args:
            threshold: 差分の閾値
            min_length: 検出する最小シワ長さ（ピクセル）
        """
        self.threshold = threshold
        self.min_length = min_length

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
        基本的なシワ検出

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

        # グレースケール変換
        gray_coax = cv2.cvtColor(img_coaxial, cv2.COLOR_BGR2GRAY)
        gray_top = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)

        debug_images['3_coaxial_gray'] = gray_coax
        debug_images['4_top_gray'] = gray_top

        # 差分計算（影の部分が強調される）
        diff = cv2.absdiff(gray_coax, gray_top)
        debug_images['5_difference'] = diff

        # より詳細な差分解析
        # 同軸照明の方が明るい部分（反射）
        reflection = np.where(gray_coax > gray_top, gray_coax - gray_top, 0).astype(np.uint8)
        # 上方照明の方が明るい部分（通常は少ない）
        shadow = np.where(gray_top > gray_coax, gray_top - gray_coax, 0).astype(np.uint8)

        debug_images['6_reflection'] = reflection
        debug_images['7_shadow'] = shadow

        # シワは主に差分として現れる
        _, wrinkle_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        debug_images['8_threshold'] = wrinkle_mask

        # ノイズ除去（横長のカーネルで横シワを強調）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 横長
        wrinkle_mask = cv2.morphologyEx(wrinkle_mask, cv2.MORPH_CLOSE, kernel)
        debug_images['9_morphology'] = wrinkle_mask

        return wrinkle_mask, diff, debug_images

    def detect_with_horizontal_emphasis(self, img_coaxial, img_top):
        """
        横シワを強調した高度な検出

        Args:
            img_coaxial: 同軸照明画像
            img_top: 上方照明画像

        Returns:
            horizontal_features: 横シワ特徴画像
            diff_normalized: 正規化済み差分画像
            debug_images: デバッグ用画像の辞書
        """
        debug_images = {}

        # 元画像を保存
        debug_images['1_coaxial_original'] = img_coaxial.copy()
        debug_images['2_top_original'] = img_top.copy()

        # 位置合わせ
        coax_aligned, top_aligned = self.align_images(img_coaxial, img_top)

        # グレースケール変換
        gray_coax = cv2.cvtColor(coax_aligned, cv2.COLOR_BGR2GRAY)
        gray_top = cv2.cvtColor(top_aligned, cv2.COLOR_BGR2GRAY)

        # 正規化（照明の総量を合わせる）
        coax_norm = cv2.normalize(gray_coax, None, 0, 255, cv2.NORM_MINMAX)
        top_norm = cv2.normalize(gray_top, None, 0, 255, cv2.NORM_MINMAX)

        debug_images['3_coaxial_normalized'] = coax_norm
        debug_images['4_top_normalized'] = top_norm

        # 差分計算（符号付き）
        diff = coax_norm.astype(float) - top_norm.astype(float)
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        debug_images['5_difference'] = diff_normalized

        # 横シワの特徴を強調
        # Sobelフィルタ（縦方向）で横エッジを検出
        sobel_y = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y_abs = np.abs(sobel_y)
        sobel_y_norm = cv2.normalize(sobel_y_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        debug_images['6_sobel_vertical'] = sobel_y_norm

        # 閾値処理
        _, sobel_binary = cv2.threshold(sobel_y_norm, 10, 255, cv2.THRESH_BINARY)
        debug_images['7_sobel_threshold'] = sobel_binary

        # 横方向に連続性があるものを抽出
        kernel_h = np.ones((1, self.min_length), np.uint8)
        horizontal_features = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel_h)

        debug_images['8_horizontal_features'] = horizontal_features

        # さらにノイズ除去
        kernel_clean = np.ones((3, 3), np.uint8)
        horizontal_features = cv2.morphologyEx(horizontal_features, cv2.MORPH_OPEN, kernel_clean)

        debug_images['9_final_wrinkles'] = horizontal_features

        return horizontal_features, diff_normalized, debug_images

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
