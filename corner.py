import cv2
import numpy as np
from PIL import Image

# 元の画像を読み込む
original_image = cv2.imread('test_source.png')  # 元のカラー画像を読み込む
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # グレースケール画像に変換

# 二値化処理（閾値を127に設定）
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Harrisコーナー検出
binary_image_float = np.float32(binary_image)
corner_response = cv2.cornerHarris(binary_image_float, 2, 3, 0.04)

# コーナーを強調表示（膨張処理）
corner_response = cv2.dilate(corner_response, None)

# コーナーの位置を緑色で元画像に描画
corner_image = original_image.copy()

# コーナーの位置を緑色に設定 (緑色はBGR形式で [0, 255, 0])
corner_image[corner_response > 0.01 * corner_response.max()] = [0, 255, 0]

# コーナー点をリストに追加
corner_points = np.argwhere(corner_response > 0.01 * corner_response.max())
corner_points = np.flip(corner_points, axis=1)  # (y, x) を (x, y) に変換

# 各コーナー点に対して水平方向と垂直方向に線を描画
for point in corner_points:
    x, y = point[0], point[1]
    
    # 水平方向の最も近い x 座標を計算
    if x < original_image.shape[1] / 2:
        closest_x = 0  # 左端
    else:
        closest_x = original_image.shape[1]  # 右端
    
    # 垂直方向の最も近い y 座標を計算
    if y < original_image.shape[0] / 2:
        closest_y = 0  # 上端
    else:
        closest_y = original_image.shape[0]  # 下端
    
    # 水平方向の線を描画 (開始位置をコーナー点に変更)
    cv2.line(corner_image, (x, y), (closest_x, y), (0, 0, 255), 2)  # 赤色の線 (BGR形式)
    
    # 垂直方向の線を描画 (開始位置をコーナー点に変更)
    cv2.line(corner_image, (x, y), (x, closest_y), (0, 0, 255), 2)  # 赤色の線 (BGR形式)

# 結果の画像を表示
corner_image_pil = Image.fromarray(corner_image)
corner_image_pil.show()

# コーナー検出後の画像を保存
corner_image_pil.save('corner_with_directional_lines.jpg')
