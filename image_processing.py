# -*- coding: utf-8 -*-
"""
画像処理モジュール
シワ検出アルゴリズムの実装
"""

import cv2
import numpy as np
from config import DETECTION_PARAMS

# YOLOモデルのグローバル変数（初回のみロード）
_yolo_model = None


def get_yolo_model():
    """
    YOLOモデルを取得（シングルトンパターン）

    Returns:
        YOLOモデル
    """
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print("YOLOモデルをロード中...")
            _yolo_model = YOLO('yolov8n.pt')  # 最軽量モデル
            print("YOLOモデルのロード完了")
        except Exception as e:
            print(f"YOLOモデルのロードに失敗: {e}")
            _yolo_model = None
    return _yolo_model


def detect_bottle_with_yolo(image):
    """
    YOLOでボトルを検出

    Args:
        image: 入力画像

    Returns:
        bottle_boxes: ボトルのバウンディングボックスのリスト [(x1, y1, x2, y2), ...]
        annotated_image: YOLO検出結果を描画した画像
    """
    model = get_yolo_model()

    if model is None:
        return [], image.copy()

    try:
        # YOLO推論（verbose=Falseでログを抑制）
        results = model(image, verbose=False)

        bottle_boxes = []
        annotated_image = image.copy()

        # 検出結果を処理
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # クラスID取得（39 = bottle）
                cls = int(box.cls[0])
                if cls == 39:  # bottle
                    # バウンディングボックス取得（x1, y1, x2, y2形式）
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])

                    bottle_boxes.append((x1, y1, x2, y2))

                    # 検出結果を描画（黄色の枠）
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    # 信頼度を表示
                    label = f"Bottle {confidence:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return bottle_boxes, annotated_image

    except Exception as e:
        print(f"YOLO検出エラー: {e}")
        return [], image.copy()


def detect_wrinkles(image):
    """
    シワ検出処理（Phase 1: 基本アルゴリズム）

    トップライト露出オーバー手法:
    1. グレースケール変換
    2. 2値化（明るい部分を除去し、影=シワを強調）
    3. ノイズ除去（モルフォロジー処理）
    4. エッジ検出（Canny）
    5. 線検出（Hough変換）
    6. 判定（検出線数でOK/NG判定）

    Args:
        image: 入力画像（BGR形式）

    Returns:
        result: "OK" or "NG"
        count: 検出された線の数
        debug_images: デバッグ用画像の辞書
    """
    debug_images = {}

    # 1. グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_images['1_grayscale'] = gray

    # 2. 2値化（露出オーバーで明るい部分を除去、暗い部分=影=シワを強調）
    threshold_value = DETECTION_PARAMS['binary_threshold']
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    debug_images['2_binary'] = binary

    # 3. ノイズ除去（モルフォロジー処理）
    kernel_size = DETECTION_PARAMS['morph_kernel_size']
    kernel = np.ones(kernel_size, np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    debug_images['3_cleaned'] = cleaned

    # 4. エッジ検出（Canny）
    low_threshold = DETECTION_PARAMS['canny_low_threshold']
    high_threshold = DETECTION_PARAMS['canny_high_threshold']
    edges = cv2.Canny(cleaned, low_threshold, high_threshold)
    debug_images['4_edges'] = edges

    # 5. 線検出（Hough変換）
    hough_threshold = DETECTION_PARAMS['hough_threshold']
    min_line_length = DETECTION_PARAMS['hough_min_line_length']
    max_line_gap = DETECTION_PARAMS['hough_max_line_gap']

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # 検出された線を描画（デバッグ用）
    line_image = image.copy()
    line_count = 0

    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    debug_images['5_detected_lines'] = line_image

    # 6. YOLOでボトル検出
    yolo_boxes, yolo_image = detect_bottle_with_yolo(image)
    debug_images['6_yolo_detection'] = yolo_image

    # 7. ボトルの実際の形状を抽出（マスク作成）
    bottle_mask, mask_visual = extract_bottle_mask(image, yolo_boxes)
    debug_images['7_bottle_mask'] = mask_visual

    # 8. ボトルをベタ塗り（マスクを使用してラベルの模様を消去）
    filled_image = create_filled_bottle_image(image, bottle_mask)
    # カラー表示用に3チャンネルに変換
    filled_image_color = cv2.cvtColor(filled_image, cv2.COLOR_GRAY2BGR)
    # YOLOの枠を描画
    if yolo_boxes:
        for box in yolo_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(filled_image_color, (x1, y1), (x2, y2), (0, 255, 255), 2)
    debug_images['8_filled_bottle'] = filled_image_color

    # 9. 輪郭検出（メイン判定、塗りつぶし画像で実行）
    contour_ng = False
    deviation_score = 0.0

    if DETECTION_PARAMS.get('use_contour_detection', True):
        contour_ng, deviation_score, contour_image = detect_bottle_contour(filled_image, image, yolo_boxes)
        debug_images['9_contour_analysis'] = contour_image
    else:
        debug_images['9_contour_analysis'] = image.copy()

    # 10. 最終判定（輪郭検出のみ）
    # 注：線検出は参考情報として残すが、判定には使用しない
    if contour_ng:
        result = "NG"
    else:
        result = "OK"

    # メタ情報を保存
    debug_images['_contour_deviation'] = deviation_score
    debug_images['_line_count'] = line_count  # 参考情報

    return result, line_count, debug_images


def draw_result_on_image(image, result, count):
    """
    検査結果を画像に描画

    Args:
        image: 元画像
        result: "OK" or "NG"
        count: 検出された線の数

    Returns:
        結果が描画された画像
    """
    result_image = image.copy()

    # 結果テキストの設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3

    # OKは緑、NGは赤
    color = (0, 255, 0) if result == "OK" else (0, 0, 255)

    # テキストを描画
    text = f"{result} (Lines: {count})"
    position = (50, 100)

    cv2.putText(result_image, text, position, font, font_scale, color, thickness)

    return result_image


def extract_bottle_mask(image, yolo_boxes):
    """
    YOLOで検出した範囲内からボトルの実際の形状を抽出

    Args:
        image: 元画像
        yolo_boxes: YOLOで検出したボトルの範囲 [(x1, y1, x2, y2), ...]

    Returns:
        mask: ボトルのマスク画像（白=ボトル、黒=背景）
        mask_visual: マスクの可視化画像（カラー）
    """
    # マスク画像を初期化（全て黒）
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # グレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if not yolo_boxes or len(yolo_boxes) == 0:
        return mask, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 最大のボトルを処理
    box = max(yolo_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = box

    # ROI範囲を切り出し
    roi_gray = gray[y1:y2, x1:x2]

    # 複数の手法でボトルを検出

    # 方法1: Cannyエッジ検出
    canny_low = DETECTION_PARAMS.get('canny_low_threshold_bottle', 30)
    canny_high = DETECTION_PARAMS.get('canny_high_threshold_bottle', 150)
    edges = cv2.Canny(roi_gray, canny_low, canny_high)

    # 方法2: Otsuの2値化
    _, otsu_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_edges = cv2.Canny(otsu_binary, 50, 150)

    # 方法3: 適応的2値化
    adaptive_binary = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    adaptive_edges = cv2.Canny(adaptive_binary, 50, 150)

    # 3つの手法を統合（OR結合）
    combined_edges = cv2.bitwise_or(edges, otsu_edges)
    combined_edges = cv2.bitwise_or(combined_edges, adaptive_edges)

    # エッジを閉じる（モルフォロジー処理）
    kernel_size = DETECTION_PARAMS.get('morph_kernel_size_bottle', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_CLOSE, kernel)  # 2回実行

    # 輪郭を検出
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 最大の輪郭を取得（ボトル本体と仮定）
        largest_contour = max(contours, key=cv2.contourArea)

        # 凸包を使用して輪郭の穴を埋める
        if DETECTION_PARAMS.get('use_convex_hull', True):
            largest_contour = cv2.convexHull(largest_contour)

        # ROI内のマスクを作成
        roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.drawContours(roi_mask, [largest_contour], -1, 255, -1)  # 塗りつぶし

        # 全体のマスクに配置
        mask[y1:y2, x1:x2] = roi_mask

    # 可視化用のマスク画像
    mask_visual = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # ボトル部分を緑で表示
    mask_visual[mask > 0] = [0, 255, 0]
    # YOLOの枠を描画
    cv2.rectangle(mask_visual, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return mask, mask_visual


def create_filled_bottle_image(image, mask):
    """
    マスクを使ってボトル範囲を一色で塗りつぶす

    Args:
        image: 元画像
        mask: ボトルのマスク画像（白=ボトル、黒=背景）

    Returns:
        filled_image: ボトルを塗りつぶした画像（グレースケール）
    """
    # グレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 塗りつぶし画像を作成
    filled = gray.copy()

    # マスク領域を中間グレー（128）で塗りつぶす
    filled[mask > 0] = 128

    return filled


def detect_bottle_contour(filled_image, original_image, yolo_boxes=None):
    """
    ボトルの輪郭を検出し、波打ち度を評価

    Args:
        filled_image: ボトルを塗りつぶした画像
        original_image: 元画像（描画用）
        yolo_boxes: YOLOで検出したボトルの範囲 [(x1, y1, x2, y2), ...]

    Returns:
        contour_ng: 輪郭が異常か（True/False）
        deviation_score: 輪郭の変動スコア
        contour_image: 輪郭を描画した画像
    """
    contour_image = original_image.copy()

    # YOLOでボトルが検出されていない場合はスキップ
    if not yolo_boxes or len(yolo_boxes) == 0:
        cv2.putText(contour_image, "Bottle not detected by YOLO", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return False, 0.0, contour_image

    # ボトル範囲を描画（シアンの枠）
    box = max(yolo_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = box
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # 2値化（グレー128を閾値にボトルと背景を分離）
    _, bottle_binary = cv2.threshold(filled_image, 100, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    bottle_binary = cv2.morphologyEx(bottle_binary, cv2.MORPH_CLOSE, kernel)
    bottle_binary = cv2.morphologyEx(bottle_binary, cv2.MORPH_OPEN, kernel)

    # 輪郭を検出
    contours, _ = cv2.findContours(bottle_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0.0, contour_image

    # 最大の輪郭を取得（ボトル本体と仮定）
    largest_contour = max(contours, key=cv2.contourArea)

    # 輪郭の面積が小さすぎる場合はスキップ
    if cv2.contourArea(largest_contour) < 1000:
        return False, 0.0, contour_image

    # 輪郭を描画（緑色）
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    # 輪郭の左右のエッジを抽出
    left_edge, right_edge = extract_left_right_edges(largest_contour)

    if left_edge is None or right_edge is None:
        return False, 0.0, contour_image

    # 左右のエッジの波打ち度を評価
    left_deviation = calculate_edge_deviation(left_edge)
    right_deviation = calculate_edge_deviation(right_edge)

    # 最大の変動を採用
    deviation_score = max(left_deviation, right_deviation)

    # 理想的な直線を描画（青色）
    draw_ideal_lines(contour_image, left_edge, right_edge)

    # 変動が大きい部分を赤色で強調
    draw_deviation_highlights(contour_image, left_edge, right_edge, deviation_score)

    # 閾値を超えたらNG
    threshold = DETECTION_PARAMS.get('contour_deviation_threshold', 10.0)
    contour_ng = deviation_score > threshold

    return contour_ng, deviation_score, contour_image


def extract_left_right_edges(contour):
    """
    輪郭から左右のエッジを抽出

    Args:
        contour: OpenCVの輪郭データ

    Returns:
        left_edge: 左エッジの点のリスト [(x, y), ...]
        right_edge: 右エッジの点のリスト [(x, y), ...]
    """
    # 輪郭を配列に変換
    points = contour.reshape(-1, 2)

    if len(points) < 4:
        return None, None

    # バウンディングボックスを取得
    x, y, w, h = cv2.boundingRect(contour)

    # 中心のX座標
    center_x = x + w // 2

    # 左右に分ける
    left_points = points[points[:, 0] < center_x]
    right_points = points[points[:, 0] >= center_x]

    if len(left_points) < 2 or len(right_points) < 2:
        return None, None

    # Y座標でソート
    left_edge = left_points[np.argsort(left_points[:, 1])]
    right_edge = right_points[np.argsort(right_points[:, 1])]

    return left_edge, right_edge


def calculate_edge_deviation(edge_points):
    """
    エッジの波打ち度（変動）を計算

    Args:
        edge_points: エッジの点のリスト

    Returns:
        deviation: 変動スコア（標準偏差）
    """
    if len(edge_points) < 10:
        return 0.0

    # X座標のみ抽出
    x_coords = edge_points[:, 0].astype(float)

    # 移動平均で平滑化
    window_size = min(DETECTION_PARAMS.get('contour_smoothing_window', 50), len(x_coords) // 2)
    if window_size < 3:
        window_size = 3

    smoothed = np.convolve(x_coords, np.ones(window_size) / window_size, mode='valid')

    # 元の座標との差分
    # smoothedの長さに合わせて元の配列を調整
    offset = (len(x_coords) - len(smoothed)) // 2
    original_subset = x_coords[offset:offset + len(smoothed)]

    deviations = np.abs(original_subset - smoothed)

    # 標準偏差を計算
    deviation_score = np.std(deviations)

    return deviation_score


def draw_ideal_lines(image, left_edge, right_edge):
    """
    理想的な直線を描画

    Args:
        image: 描画対象の画像
        left_edge: 左エッジ
        right_edge: 右エッジ
    """
    for edge in [left_edge, right_edge]:
        if edge is None or len(edge) < 2:
            continue

        # 最小二乗法で直線をフィッティング
        y_coords = edge[:, 1]
        x_coords = edge[:, 0]

        # 直線の係数を計算
        coeffs = np.polyfit(y_coords, x_coords, 1)
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # 直線を描画（青色）
        x_start = int(np.polyval(coeffs, y_min))
        x_end = int(np.polyval(coeffs, y_max))

        cv2.line(image, (x_start, y_min), (x_end, y_max), (255, 0, 0), 2)


def draw_deviation_highlights(image, left_edge, right_edge, deviation_score):
    """
    変動が大きい部分を強調表示

    Args:
        image: 描画対象の画像
        left_edge: 左エッジ
        right_edge: 右エッジ
        deviation_score: 変動スコア
    """
    threshold = DETECTION_PARAMS.get('contour_deviation_threshold', 10.0)

    if deviation_score <= threshold:
        return

    # 変動が大きい場合、エッジを赤色で強調
    for edge in [left_edge, right_edge]:
        if edge is None or len(edge) < 2:
            continue

        # エッジ上に小さな円を描画
        for point in edge[::5]:  # 5点ごとに描画
            cv2.circle(image, tuple(point), 3, (0, 0, 255), -1)
