from PIL import Image
import cv2
import numpy as np

# 画像を読み込む
image = Image.open('test_source.png')

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

# ガウシアンブラーでノイズ除去
blurred_image = cv2.GaussianBlur(opencv_image, (5, 5), 0)

# 二値化処理（黒い線部分を検出）
_, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)  # 黒い部分を白として扱う

# 二値化後の画像をカラーに変換して緑色の点を描画できるようにする
color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# 黒い部分（0の部分）を緑で描画
color_image[binary_image == 255] = [0, 255, 0]  # 緑色 (BGR形式で [0, 255, 0])

# 結果の画像をPIL形式に変換
final_image = Image.fromarray(color_image)

# 二値化後の画像を保存
final_image.save('test_with_green_lines.jpg')

# 画像を表示
final_image.show()
