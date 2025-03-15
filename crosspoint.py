import cv2
import numpy as np
from PIL import Image

# グレースケール画像を読み込む
image = Image.open('resized_grayscale_image.jpg')

# Pillow画像をOpenCV形式に変換（RGBからグレースケールに変換済み）
opencv_image = np.array(image)

# グレースケール画像をカラー画像に変換（赤い点を描画するため）
if len(opencv_image.shape) == 2:  # グレースケールの場合
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)

# グレー領域の閾値を定義（グレー領域を検出するため）
lower_gray = np.array([130, 130, 130])  # 最小グレースケール値
upper_gray = np.array([210, 210, 210])  # 最大グレースケール値

# グレー領域をマスク
gray_mask = cv2.inRange(opencv_image, lower_gray, upper_gray)

# Harrisコーナー検出を使って交差点（角）を検出
corners = cv2.cornerHarris(opencv_image[:, :, 0], blockSize=2, ksize=3, k=0.04)

# コーナーがある場所に赤い点を描画
corners = cv2.dilate(corners, None)  # コーナーを膨張させる
threshold = 0.01 * corners.max()
corner_points = np.argwhere(corners > threshold)  # コーナーの位置を取得

# 点と点を結ぶために線を引く（水平と垂直方向）
# 水平線と垂直線を引くために、各点について同じxまたはy座標を持つ点を検索
for i in range(len(corner_points)):
    point1 = (corner_points[i][1], corner_points[i][0])  # (x, y) に変換

    # コーナーがグレー領域内にあるかを確認
    if gray_mask[point1[1], point1[0]] == 0:  # グレー領域外の点は除外
        continue

    # 水平線を引く: 同じy座標を持つ他の点を検索
    for j in range(len(corner_points)):
        if i != j:  # 同じ点は除外
            point2 = (corner_points[j][1], corner_points[j][0])  # (x, y) に変換
            if point1[1] == point2[1]:  # y座標が同じ（水平）
                if gray_mask[point2[1], point2[0]] == 0:  # グレー領域外の点は除外
                    continue
                # 両方の点がグレー領域内であれば線を引く
                cv2.line(opencv_image, point1, point2, (0, 255, 0), 1)  # 緑色の線を描画

    # 垂直線を引く: 同じx座標を持つ他の点を検索
    for j in range(len(corner_points)):
        if i != j:  # 同じ点は除外
            point2 = (corner_points[j][1], corner_points[j][0])  # (x, y) に変換
            if point1[0] == point2[0]:  # x座標が同じ（垂直）
                if gray_mask[point2[1], point2[0]] == 0:  # グレー領域外の点は除外
                    continue
                # 両方の点がグレー領域内であれば線を引く
                cv2.line(opencv_image, point1, point2, (0, 255, 0), 1)  # 緑色の線を描画

# コーナーに赤い点を描画
for point in corner_points:
    cv2.circle(opencv_image, (point[1], point[0]), 5, (0, 0, 255), -1)  # 赤い点

# 結果を表示
final_image = Image.fromarray(opencv_image)
final_image.show()

# 最終的な画像を保存
final_image.save('output_image_with_filtered_lines.jpg')
