# -*- coding: utf-8 -*-
"""
カメラ制御モジュール
オムロンセンテック STC-MCS231U3V用
OpenCVバックエンドを使用した安定したカメラ制御
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple
from config import CAMERA_SETTINGS

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StCameraControl:
    """
    カメラ制御クラス

    OpenCVを使用して複数のバックエンドに対応
    カメラインデックスの自動検出機能付き
    """

    def __init__(self):
        """初期化"""
        self.camera = None
        self.is_opened = False
        self.current_camera_index = None
        self.current_backend = None
        self.frame_lock = threading.Lock()
        self.current_frame = None

        # 試行するバックエンドのリスト（優先順）
        self.backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto")
        ]

        # 試行するカメラインデックスの範囲
        self.max_camera_index = 5

    def open(self, camera_index: Optional[int] = None) -> bool:
        """
        カメラをオープン

        Args:
            camera_index: カメラ番号（Noneの場合は自動検出）

        Returns:
            成功: True, 失敗: False
        """
        if camera_index is not None:
            # 指定されたインデックスのカメラを開く
            return self._open_camera_with_index(camera_index)
        else:
            # カメラを自動検出
            return self._auto_detect_camera()

    def _open_camera_with_index(self, camera_index: int) -> bool:
        """
        指定されたインデックスのカメラを開く

        Args:
            camera_index: カメラ番号

        Returns:
            成功: True, 失敗: False
        """
        # 既存のカメラを完全に解放
        if self.camera:
            self.camera.release()
            self.camera = None
            time.sleep(0.3)

        # 各バックエンドを試行
        for backend, backend_name in self.backends:
            logger.info(f"カメラ {camera_index} をバックエンド {backend_name} で接続試行中...")

            try:
                camera = cv2.VideoCapture(camera_index, backend)

                if camera.isOpened():
                    # カメラ設定を適用
                    if self._configure_camera(camera):
                        # テストフレームを取得
                        if self._test_frame_capture(camera):
                            self.camera = camera
                            self.is_opened = True
                            self.current_camera_index = camera_index
                            self.current_backend = backend_name

                            # 設定情報を表示
                            self._log_camera_info()

                            logger.info(f"✓ カメラ {camera_index} を {backend_name} で接続成功")
                            return True

                # 失敗した場合はカメラを解放
                camera.release()

            except Exception as e:
                logger.warning(f"バックエンド {backend_name} でエラー: {e}")
                continue

        logger.error(f"カメラ {camera_index} の接続に失敗しました")
        return False

    def _auto_detect_camera(self) -> bool:
        """
        カメラを自動検出

        Returns:
            成功: True, 失敗: False
        """
        logger.info("カメラの自動検出を開始...")

        # インデックス0から順番に試行
        for index in range(self.max_camera_index + 1):
            if self._open_camera_with_index(index):
                logger.info(f"カメラをインデックス {index} で検出しました")
                return True

        logger.error("使用可能なカメラが見つかりませんでした")
        return False

    def _configure_camera(self, camera) -> bool:
        """
        カメラ設定を適用

        Args:
            camera: OpenCVカメラオブジェクト

        Returns:
            成功: True, 失敗: False
        """
        try:
            # 解像度設定
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])

            # FPS設定
            camera.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])

            # バッファサイズを最小化（レイテンシ削減）
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 自動露出を完全に無効化
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 0 = 完全マニュアル

            # 初期露出値を設定（絶対値で試行）
            camera.set(cv2.CAP_PROP_EXPOSURE, CAMERA_SETTINGS['exposure_time'])

            # それでもダメなら対数値
            exposure_value = -int(np.log2(CAMERA_SETTINGS['exposure_time'] / 1000000.0))
            camera.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

            # ゲイン設定
            camera.set(cv2.CAP_PROP_GAIN, CAMERA_SETTINGS['gain'])

            # 明るさ設定
            camera.set(cv2.CAP_PROP_BRIGHTNESS, CAMERA_SETTINGS['brightness'])

            return True

        except Exception as e:
            logger.error(f"カメラ設定エラー: {e}")
            return False

    def _test_frame_capture(self, camera, max_retries: int = 5) -> bool:
        """
        テストフレームを取得

        Args:
            camera: OpenCVカメラオブジェクト
            max_retries: 最大リトライ回数

        Returns:
            成功: True, 失敗: False
        """
        for i in range(max_retries):
            ret, frame = camera.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
                logger.info(f"テストフレーム取得成功 (試行 {i+1}/{max_retries})")
                return True

            # 少し待機してリトライ
            time.sleep(0.2)
            logger.warning(f"フレーム取得失敗 (試行 {i+1}/{max_retries})")

        logger.error("テストフレームの取得に失敗")
        return False

    def _log_camera_info(self):
        """カメラ情報をログ出力"""
        if self.camera and self.camera.isOpened():
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

            # カメラのプロパティサポート状況を確認
            exposure = self.camera.get(cv2.CAP_PROP_EXPOSURE)
            brightness = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
            gain = self.camera.get(cv2.CAP_PROP_GAIN)
            auto_exposure = self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)

            logger.info("=" * 50)
            logger.info(f"カメラインデックス: {self.current_camera_index}")
            logger.info(f"バックエンド: {self.current_backend}")
            logger.info(f"解像度: {actual_width}x{actual_height}")
            logger.info(f"FPS: {actual_fps}")
            logger.info(f"現在の露出: {exposure}")
            logger.info(f"現在の明るさ: {brightness}")
            logger.info(f"現在のゲイン: {gain}")
            logger.info(f"自動露出: {auto_exposure}")
            logger.info("=" * 50)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        1フレームをキャプチャ

        Returns:
            成功: 画像(numpy array), 失敗: None
        """
        if not self.is_opened or not self.camera:
            logger.warning("カメラが開かれていません")
            return None

        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                return frame
            else:
                logger.warning("フレーム取得失敗")
                return None

        except Exception as e:
            logger.error(f"フレームキャプチャエラー: {e}")
            return None

    def get_frame(self) -> Optional[np.ndarray]:
        """
        現在のフレームを取得（スレッドセーフ）

        Returns:
            成功: 画像(numpy array), 失敗: None
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def set_exposure(self, exposure_time: int):
        """
        露出時間を設定

        Args:
            exposure_time: 露出時間（マイクロ秒）
        """
        if not self.is_opened or not self.camera:
            logger.warning("カメラが開かれていません")
            return

        try:
            # 複数の方法で露出設定を試行

            # 方法1: 絶対値で設定
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_time)
            actual1 = self.camera.get(cv2.CAP_PROP_EXPOSURE)

            # 方法2: 対数値で設定
            exposure_value = -int(np.log2(exposure_time / 1000000.0))
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
            actual2 = self.camera.get(cv2.CAP_PROP_EXPOSURE)

            # 方法3: ミリ秒単位
            exposure_ms = exposure_time / 1000.0
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_ms)
            actual3 = self.camera.get(cv2.CAP_PROP_EXPOSURE)

            logger.info(f"露出設定: {exposure_time} μs (絶対値:{actual1}, 対数:{actual2}, ms:{actual3})")

        except Exception as e:
            logger.error(f"露出設定エラー: {e}")

    def set_gain(self, gain: float):
        """
        ゲインを設定

        Args:
            gain: ゲイン値（dB）
        """
        if not self.is_opened or not self.camera:
            logger.warning("カメラが開かれていません")
            return

        try:
            self.camera.set(cv2.CAP_PROP_GAIN, gain)

            # 実際に設定された値を確認
            actual_value = self.camera.get(cv2.CAP_PROP_GAIN)
            logger.info(f"ゲイン設定: {gain} dB (実際: {actual_value})")

        except Exception as e:
            logger.error(f"ゲイン設定エラー: {e}")

    def set_brightness(self, brightness: int):
        """
        明るさを設定

        Args:
            brightness: 明るさ値（0-255）
        """
        if not self.is_opened or not self.camera:
            logger.warning("カメラが開かれていません")
            return

        try:
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

            # 実際に設定された値を確認
            actual_value = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
            logger.info(f"明るさ設定: {brightness} (実際: {actual_value})")

        except Exception as e:
            logger.error(f"明るさ設定エラー: {e}")

    def get_camera_info(self) -> dict:
        """
        カメラ情報を取得

        Returns:
            カメラ情報の辞書
        """
        if self.camera and self.is_opened:
            return {
                'index': self.current_camera_index,
                'backend': self.current_backend,
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE),
                'gain': self.camera.get(cv2.CAP_PROP_GAIN),
                'is_opened': self.is_opened
            }
        return {'error': 'Camera not initialized'}

    def close(self):
        """カメラをクローズ"""
        self.is_opened = False

        if self.camera:
            self.camera.release()
            self.camera = None
            logger.info("カメラをクローズしました")

        self.current_camera_index = None
        self.current_backend = None

        with self.frame_lock:
            self.current_frame = None

    def scan_available_cameras(self) -> list:
        """
        利用可能なカメラをスキャン

        Returns:
            利用可能なカメラ情報のリスト
            [{
                'index': int,
                'backend': str,
                'width': int,
                'height': int,
                'name': str
            }, ...]
        """
        available_cameras = []
        logger.info("利用可能なカメラをスキャン中...")

        for index in range(self.max_camera_index + 1):
            for backend, backend_name in self.backends:
                try:
                    camera = cv2.VideoCapture(index, backend)

                    if camera.isOpened():
                        # テストフレームを取得
                        ret, frame = camera.read()
                        if ret and frame is not None:
                            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            camera_info = {
                                'index': index,
                                'backend': backend_name,
                                'width': width,
                                'height': height,
                                'name': f"カメラ {index} ({backend_name}, {width}x{height})"
                            }
                            available_cameras.append(camera_info)
                            logger.info(f"検出: {camera_info['name']}")

                            camera.release()
                            time.sleep(0.2)
                            break  # 最初に成功したバックエンドで確定

                    camera.release()

                except Exception as e:
                    logger.debug(f"カメラ {index} (バックエンド {backend_name}): {e}")
                    continue

        logger.info(f"スキャン完了: {len(available_cameras)}台のカメラを検出")
        return available_cameras

    def switch_camera(self, camera_index: int) -> bool:
        """
        カメラを切り替え

        Args:
            camera_index: 切り替え先のカメラ番号

        Returns:
            成功: True, 失敗: False
        """
        logger.info(f"カメラを {self.current_camera_index} から {camera_index} に切り替え")

        # 現在のカメラを閉じる
        was_opened = self.is_opened
        if was_opened:
            self.close()
            time.sleep(0.5)  # リソース解放を待つ

        # 新しいカメラを開く
        if self._open_camera_with_index(camera_index):
            logger.info(f"カメラ {camera_index} への切り替え成功")
            return True
        else:
            logger.error(f"カメラ {camera_index} への切り替え失敗")
            return False

    def __del__(self):
        """デストラクタ"""
        self.close()
