# -*- coding: utf-8 -*-
"""
照明制御モジュール
同軸照明と上方照明の制御
"""

import logging
import time

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightingControl:
    """
    照明制御クラス

    実際のハードウェア接続時はGPIO制御等に対応
    現在はシミュレーションモード
    """

    def __init__(self, use_gpio=False):
        """
        初期化

        Args:
            use_gpio: GPIO制御を使用するか（Raspberry Pi等）
        """
        self.use_gpio = use_gpio
        self.coaxial_state = False
        self.top_state = False

        if use_gpio:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.GPIO.setmode(GPIO.BCM)

                # GPIOピン設定（実環境に合わせて変更）
                self.COAXIAL_PIN = 17
                self.TOP_PIN = 27

                self.GPIO.setup(self.COAXIAL_PIN, GPIO.OUT)
                self.GPIO.setup(self.TOP_PIN, GPIO.OUT)

                logger.info("GPIO制御を初期化しました")
            except ImportError:
                logger.warning("RPi.GPIOが見つかりません。シミュレーションモードで動作します")
                self.use_gpio = False
        else:
            logger.info("シミュレーションモードで動作します")

    def set_coaxial_light(self, state: bool):
        """
        同軸照明を制御

        Args:
            state: True=ON, False=OFF
        """
        self.coaxial_state = state

        if self.use_gpio:
            self.GPIO.output(self.COAXIAL_PIN, GPIO.HIGH if state else GPIO.LOW)
            logger.info(f"同軸照明: {'ON' if state else 'OFF'}")
        else:
            logger.info(f"[シミュレーション] 同軸照明: {'ON' if state else 'OFF'}")

    def set_top_light(self, state: bool):
        """
        上方照明を制御

        Args:
            state: True=ON, False=OFF
        """
        self.top_state = state

        if self.use_gpio:
            self.GPIO.output(self.TOP_PIN, GPIO.HIGH if state else GPIO.LOW)
            logger.info(f"上方照明: {'ON' if state else 'OFF'}")
        else:
            logger.info(f"[シミュレーション] 上方照明: {'ON' if state else 'OFF'}")

    def all_off(self):
        """全ての照明をOFF"""
        self.set_coaxial_light(False)
        self.set_top_light(False)

    def capture_sequence(self, camera, stabilization_time=0.1):
        """
        照明切り替えシーケンスで撮影

        Args:
            camera: カメラオブジェクト
            stabilization_time: 照明切り替え後の安定待ち時間（秒）

        Returns:
            dict: {'coaxial': 同軸照明画像, 'top': 上方照明画像}
        """
        images = {}

        # 1. 同軸照明のみON
        logger.info("=== 同軸照明で撮影 ===")
        self.set_coaxial_light(True)
        self.set_top_light(False)
        time.sleep(stabilization_time)
        images['coaxial'] = camera.capture_frame()

        # 2. 上方照明のみON
        logger.info("=== 上方照明で撮影 ===")
        self.set_coaxial_light(False)
        self.set_top_light(True)
        time.sleep(stabilization_time)
        images['top'] = camera.capture_frame()

        # 3. 両方OFF
        self.all_off()

        return images

    def get_state(self):
        """
        現在の照明状態を取得

        Returns:
            dict: {'coaxial': bool, 'top': bool}
        """
        return {
            'coaxial': self.coaxial_state,
            'top': self.top_state
        }

    def cleanup(self):
        """終了処理"""
        self.all_off()

        if self.use_gpio:
            self.GPIO.cleanup()
            logger.info("GPIO制御をクリーンアップしました")

    def __del__(self):
        """デストラクタ"""
        self.cleanup()
