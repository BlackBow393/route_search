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

# Harrisコーナー検出を使って交差点（角）を検出
corners = cv2.cornerHarris(opencv_image[:, :, 0], blockSize=2, ksize=3, k=0.04)

# コーナーを膨張させる
corners = cv2.dilate(corners, None)

# 強いコーナーに赤い円を描画
threshold = 0.01 * corners.max()  # 閾値を設定
for y in range(corners.shape[0]):
    for x in range(corners.shape[1]):
        if corners[y, x] > threshold:  # コーナーの強さが閾値を超えている場合
            cv2.circle(opencv_image, (x, y), 5, (0, 0, 255), 2)  # 赤い円を描画

# 結果を表示
final_image = Image.fromarray(opencv_image)
final_image.show()

# 最終的な画像を保存
final_image.save('final_image_with_intersections.jpg')
