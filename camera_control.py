# -*- coding: utf-8 -*-
"""
カメラ制御モジュール（Harvesters版）
オムロンセンテック STC-MCS231U3V用
Harvesters（GenICam）を使用した完全なカメラ制御
"""

import numpy as np
import logging
from typing import Optional
from harvesters.core import Harvester
from config import CAMERA_SETTINGS

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CTIファイルのパス
CTI_FILE = r"C:\Program Files\Common Files\OMRON_SENTECH\GenTL\v1_5\StGenTL_MD_VC141_v1_5_x64.cti"


class StCameraControl:
    """
    カメラ制御クラス（Harvesters版）

    Harvesters（GenICam）を使用してSentechカメラを完全制御
    """

    def __init__(self):
        """初期化"""
        self.harvester = None
        self.image_acquirer = None
        self.is_opened = False
        self.current_camera_index = None
        self.current_backend = "Harvesters (GenICam)"
        self.bayer_pattern = "BG"  # デフォルトはBG（正しいパターン）

    def open(self, camera_index: Optional[int] = None) -> bool:
        """
        カメラをオープン

        Args:
            camera_index: カメラ番号（Harvestersでは無視、最初のカメラを使用）

        Returns:
            成功: True, 失敗: False
        """
        try:
            # Harvesterを初期化
            self.harvester = Harvester()

            # CTIファイルを追加
            self.harvester.add_file(CTI_FILE)

            # デバイス情報を更新
            self.harvester.update()

            # 利用可能なデバイスを確認
            if len(self.harvester.device_info_list) == 0:
                logger.error("利用可能なカメラが見つかりませんでした")
                return False

            # 最初のカメラを使用
            logger.info(f"カメラを検出: {len(self.harvester.device_info_list)}台")

            # Image Acquirerを作成
            self.image_acquirer = self.harvester.create(0)

            # カメラのノードマップを取得
            node_map = self.image_acquirer.remote_device.node_map

            # 初期設定
            self._configure_camera(node_map)

            # 画像取得を開始
            self.image_acquirer.start()

            self.is_opened = True
            self.current_camera_index = 0

            # ウォームアップ: 最初の数フレームを破棄してバッファを安定させる
            warmup_frames = 5
            for i in range(warmup_frames):
                try:
                    with self.image_acquirer.fetch(timeout=2.0) as buffer:
                        pass  # フレームを破棄
                except:
                    pass

            logger.info("カメラ起動完了")
            return True

        except Exception as e:
            logger.error(f"カメラのオープンに失敗: {e}")
            if self.harvester:
                self.harvester.reset()
                self.harvester = None
            return False

    def _configure_camera(self, node_map):
        """
        カメラの初期設定

        Args:
            node_map: GenICamノードマップ
        """
        try:
            # 取得モードを連続撮影に設定
            try:
                node_map.AcquisitionMode.value = "Continuous"
            except Exception as e:
                logger.warning(f"AcquisitionMode設定失敗: {e}")

            # トリガーモードを無効化（フリーラン）
            try:
                node_map.TriggerMode.value = "Off"
            except Exception as e:
                logger.warning(f"TriggerMode設定失敗: {e}")

            # フレームレート制御を無効化（最大速度で撮影）
            try:
                if hasattr(node_map, 'AcquisitionFrameRateEnable'):
                    node_map.AcquisitionFrameRateEnable.value = False
            except Exception as e:
                logger.warning(f"フレームレート設定失敗: {e}")

            # より高画質なピクセルフォーマットを選択
            try:
                available_formats = node_map.PixelFormat.symbolics
                # 12bit > 10bit > 8bit の順で試す
                if "BayerRG12" in str(available_formats):
                    node_map.PixelFormat.value = "BayerRG12"
                elif "BayerRG10" in str(available_formats):
                    node_map.PixelFormat.value = "BayerRG10"
                elif "Mono8" in str(available_formats):
                    node_map.PixelFormat.value = "Mono8"
            except Exception as e:
                logger.warning(f"ピクセルフォーマット設定失敗: {e}")

            # 画像サイズ設定
            node_map.Width.value = CAMERA_SETTINGS['width']
            node_map.Height.value = CAMERA_SETTINGS['height']

            # 露出モードを手動に設定
            node_map.ExposureAuto.value = "Off"
            node_map.ExposureMode.value = "Timed"
            node_map.ExposureTime.value = CAMERA_SETTINGS['exposure_time']

            # ゲイン設定
            node_map.GainAuto.value = "Off"
            node_map.Gain.value = CAMERA_SETTINGS['gain']

        except Exception as e:
            logger.warning(f"一部の設定に失敗: {e}")

    def scan_available_cameras(self) -> list:
        """
        利用可能なカメラをスキャン

        Returns:
            利用可能なカメラ情報のリスト
        """
        available_cameras = []

        try:
            # 一時的にHarvesterを作成
            temp_harvester = Harvester()
            temp_harvester.add_file(CTI_FILE)
            temp_harvester.update()

            for i, device_info in enumerate(temp_harvester.device_info_list):
                camera_info = {
                    'index': i,
                    'backend': 'Harvesters',
                    'width': CAMERA_SETTINGS['width'],
                    'height': CAMERA_SETTINGS['height'],
                    'name': f"カメラ {i} (Harvesters, {device_info.display_name})"
                }
                available_cameras.append(camera_info)

            temp_harvester.reset()

        except Exception as e:
            logger.error(f"カメラスキャンエラー: {e}")

        return available_cameras

    def switch_camera(self, camera_index: int) -> bool:
        """
        カメラを切り替え

        Args:
            camera_index: 切り替え先のカメラ番号

        Returns:
            成功: True, 失敗: False
        """
        logger.info(f"カメラ切り替え: {camera_index}")

        # 現在のカメラを閉じて再度開く
        self.close()
        return self.open(camera_index)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        1フレームをキャプチャ

        Returns:
            成功: 画像(numpy array), 失敗: None
        """
        if not self.is_opened or not self.image_acquirer:
            return None

        try:
            # バッファからフレームを取得
            with self.image_acquirer.fetch(timeout=2.0) as buffer:
                # PayloadをNumPy配列に変換
                if not buffer or not buffer.payload or not buffer.payload.components:
                    return None

                component = buffer.payload.components[0]

                # ピクセルフォーマットに応じて処理
                pixel_format = self.image_acquirer.remote_device.node_map.PixelFormat.value

                if "Bayer" in pixel_format:
                    # Bayer形式の場合、カラーに変換
                    import cv2

                    # データ型とサイズを確認
                    raw_data = component.data

                    # 10bit/12bitの場合、8bitに正規化
                    if "12" in pixel_format:
                        # 12bit → 8bit
                        if raw_data.dtype == np.uint16:
                            # 既に16bitの場合
                            image_16bit = raw_data.reshape(component.height, component.width)
                            image = (image_16bit >> 4).astype(np.uint8)
                        else:
                            # 12bitパックデータの場合は展開が必要
                            # とりあえず8bitとして扱う
                            image = raw_data.reshape(component.height, component.width).astype(np.uint8)

                    elif "10" in pixel_format:
                        # 10bit → 8bit
                        if raw_data.dtype == np.uint16:
                            image_16bit = raw_data.reshape(component.height, component.width)
                            image = (image_16bit >> 2).astype(np.uint8)
                        else:
                            image = raw_data.reshape(component.height, component.width).astype(np.uint8)

                    else:
                        # 8bitの場合
                        image = raw_data.reshape(component.height, component.width).astype(np.uint8)

                    # 設定されたBayerパターンで変換
                    bayer_map = {
                        "RG": cv2.COLOR_BAYER_RG2BGR,
                        "GR": cv2.COLOR_BAYER_GR2BGR,
                        "GB": cv2.COLOR_BAYER_GB2BGR,
                        "BG": cv2.COLOR_BAYER_BG2BGR
                    }

                    conversion = bayer_map.get(self.bayer_pattern, cv2.COLOR_BAYER_BG2BGR)
                    image_bgr = cv2.cvtColor(image, conversion)
                elif len(image.shape) == 2:
                    # モノクロの場合、BGRに変換
                    image_bgr = np.stack([image, image, image], axis=2)
                else:
                    # 既にカラーの場合
                    image_bgr = image

                return image_bgr

        except Exception as e:
            # エラー時は何もしない（ログ出力なし）
            return None

    def set_exposure(self, exposure_time: int):
        """
        露出時間を設定

        Args:
            exposure_time: 露出時間（マイクロ秒）
        """
        if not self.is_opened or not self.image_acquirer:
            logger.warning("カメラが開かれていません")
            return

        try:
            node_map = self.image_acquirer.remote_device.node_map
            node_map.ExposureTime.value = float(exposure_time)

            actual_value = node_map.ExposureTime.value
            logger.info(f"露出設定: {exposure_time} μs (実際: {actual_value} μs)")

        except Exception as e:
            logger.error(f"露出設定エラー: {e}")

    def set_gain(self, gain: float):
        """
        ゲインを設定

        Args:
            gain: ゲイン値（dB）
        """
        if not self.is_opened or not self.image_acquirer:
            logger.warning("カメラが開かれていません")
            return

        try:
            node_map = self.image_acquirer.remote_device.node_map
            node_map.Gain.value = float(gain)

            actual_value = node_map.Gain.value
            logger.info(f"ゲイン設定: {gain} dB (実際: {actual_value} dB)")

        except Exception as e:
            logger.error(f"ゲイン設定エラー: {e}")

    def set_brightness(self, brightness: int):
        """
        明るさを設定（Harvesters版ではゲインで代用）

        Args:
            brightness: 明るさ値（0-255）
        """
        # GenICamには明るさという概念がないため、ゲインで代用
        # 0-255 → 0-30dB に変換
        gain_value = (brightness / 255.0) * 30.0
        self.set_gain(gain_value)

    def set_gamma(self, gamma: float):
        """
        ガンマ補正を設定

        Args:
            gamma: ガンマ値（0.5-2.0）
        """
        if not self.is_opened or not self.image_acquirer:
            logger.warning("カメラが開かれていません")
            return

        try:
            node_map = self.image_acquirer.remote_device.node_map
            if hasattr(node_map, 'Gamma'):
                node_map.Gamma.value = float(gamma)
                logger.info(f"ガンマ設定: {gamma}")
            else:
                logger.warning("このカメラはガンマ設定に対応していません")
        except Exception as e:
            logger.error(f"ガンマ設定エラー: {e}")

    def set_white_balance(self, red_ratio: float, blue_ratio: float):
        """
        ホワイトバランスを設定

        Args:
            red_ratio: 赤の比率（0.5-2.0）
            blue_ratio: 青の比率（0.5-2.0）
        """
        if not self.is_opened or not self.image_acquirer:
            logger.warning("カメラが開かれていません")
            return

        try:
            node_map = self.image_acquirer.remote_device.node_map

            # ホワイトバランスを手動に設定
            if hasattr(node_map, 'BalanceWhiteAuto'):
                node_map.BalanceWhiteAuto.value = "Off"

            # 赤の比率を設定
            if hasattr(node_map, 'BalanceRatioSelector') and hasattr(node_map, 'BalanceRatio'):
                node_map.BalanceRatioSelector.value = "Red"
                node_map.BalanceRatio.value = float(red_ratio)

                # 青の比率を設定
                node_map.BalanceRatioSelector.value = "Blue"
                node_map.BalanceRatio.value = float(blue_ratio)

                logger.info(f"ホワイトバランス設定: R={red_ratio:.2f}, B={blue_ratio:.2f}")
            else:
                logger.warning("このカメラはホワイトバランス設定に対応していません")
        except Exception as e:
            logger.error(f"ホワイトバランス設定エラー: {e}")

    def set_bayer_pattern(self, pattern: str):
        """
        Bayerパターンを設定

        Args:
            pattern: "RG", "GR", "GB", "BG" のいずれか
        """
        if pattern in ["RG", "GR", "GB", "BG"]:
            self.bayer_pattern = pattern
            logger.info(f"Bayerパターンを {pattern} に設定")

    def get_camera_info(self) -> dict:
        """
        カメラ情報を取得

        Returns:
            カメラ情報の辞書
        """
        if self.image_acquirer and self.is_opened:
            try:
                node_map = self.image_acquirer.remote_device.node_map
                return {
                    'index': self.current_camera_index,
                    'backend': self.current_backend,
                    'width': node_map.Width.value,
                    'height': node_map.Height.value,
                    'exposure': node_map.ExposureTime.value,
                    'gain': node_map.Gain.value,
                    'is_opened': self.is_opened
                }
            except:
                pass

        return {'error': 'Camera not initialized'}

    def close(self):
        """カメラをクローズ"""
        try:
            if self.image_acquirer:
                self.image_acquirer.stop()
                self.image_acquirer.destroy()
                self.image_acquirer = None

            if self.harvester:
                self.harvester.reset()
                self.harvester = None

            self.is_opened = False
            self.current_camera_index = None

            logger.info("カメラをクローズしました")

        except Exception as e:
            logger.error(f"カメラクローズエラー: {e}")

    def __del__(self):
        """デストラクタ"""
        self.close()
