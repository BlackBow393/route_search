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

# コーナーを赤色で元画像に描画
# コーナーの位置を赤色に設定 (赤色はBGR形式で [0, 0, 255])
corner_image = original_image.copy()
corner_image[corner_response > 0.01 * corner_response.max()] = [0, 255, 0]  # 赤色 (BGR)

# 結果の画像を表示
corner_image_pil = Image.fromarray(corner_image)
corner_image_pil.show()

# コーナー検出後の画像を保存
corner_image_pil.save('corner_detected_on_original.jpg')
