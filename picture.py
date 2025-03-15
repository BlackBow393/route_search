from PIL import Image
import cv2
import numpy as np

# 画像を読み込む
image = Image.open('テスト1.png')

# グレースケールに変換
gray_image = image.convert('L')

# 画像のサイズを取得
width, height = gray_image.size

# 16:9 のアスペクト比に基づいて新しいサイズを計算
if width / height > 16 / 9:
    new_width = width
    new_height = int(width * 9 / 16)  # 幅に合わせて高さを調整
else:
    new_height = height
    new_width = int(height * 16 / 9)  # 高さに合わせて幅を調整

# リサイズ
resized_image = gray_image.resize((new_width, new_height))

# リサイズ後の画像をNumPy配列に変換（OpenCV用）
opencv_image = np.array(resized_image)

# 二値化処理（閾値を127に設定）
ret, binary_image = cv2.threshold(opencv_image, 127, 255, cv2.THRESH_BINARY)

# 二値化された画像をPIL形式に変換
binary_pil_image = Image.fromarray(binary_image)

# 二値化後の画像を保存
binary_pil_image.save('test.jpg')

# 画像を表示
binary_pil_image.show()
