from PIL import Image

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

# リサイズ後の画像を保存
resized_image.save('resized_grayscale_image.jpg')

# 画像を表示
resized_image.show()
