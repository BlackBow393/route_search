from PIL import Image

# 画像を読み込む
image = Image.open('テスト1.png')

# グレースケールに変換
gray_image = image.convert('L')

# 変換後の画像を保存
gray_image.save('output_image.jpg')

# 画像を表示
gray_image.show()
