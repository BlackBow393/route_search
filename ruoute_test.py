import cv2
import numpy as np
from PIL import Image

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

# 二値化後の画像をカラーに変換して描画できるようにする
color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# 黒い部分（0の部分）を白で描画
color_image[binary_image == 255] = [255, 255, 255]  # 白色 (RGB形式で [255, 255, 255])

# 座標 (60, 2400) に赤い点を描画 (RGB形式で [255, 0, 0])
cv2.circle(color_image, (60, 2400), 10, (255, 0, 0), -1)  # 半径10の赤い点を描画

# 座標 (3650, 150) にもう1つ赤い点を描画
cv2.circle(color_image, (3650, 150), 10, (255, 0, 0), -1)  # 別の位置に赤い点を描画

# 'Start' という文字を最初の赤い点の近くに描画
cv2.putText(color_image, 'Start', (60 + 10, 2400 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# 'Goal' という文字を2つ目の赤い点の近くに描画
cv2.putText(color_image, 'Goal', (3650 + 10, 150 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# StartとGoalの座標
start = (2400, 60)  # (Y, X)で指定
goal = (150, 3650)  # (Y, X)で指定

# 直線を描画し、白の領域と交差したところに赤いノードを描画する関数
def draw_line_with_intersections(start, goal, grid):
    y1, x1 = start  # (Y, X) -> (Y, X)に変換
    y2, x2 = goal   # (Y, X) -> (Y, X)に変換
    
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))  # 直線の描画に必要なステップ数
    
    increment_x = dx / steps
    increment_y = dy / steps
    
    x = x1
    y = y1
    intersection_points = []
    
    # 直線上の各点を描画
    for i in range(steps + 1):
        x_int = int(round(x))
        y_int = int(round(y))
        
        if 0 <= x_int < len(grid[0]) and 0 <= y_int < len(grid) and grid[y_int, x_int] == 255:  # 白の領域
            intersection_points.append((y_int, x_int))  # (Y, X)を追加
        
        # 次の点に移動
        x += increment_x
        y += increment_y
    
    return intersection_points

# 二値化された画像を2Dグリッドに変換
grid = np.array(binary_image)

# 直線の途中で衝突した点をStartのX軸に合わせて描画する
def adjust_point_to_start_x(intersection_point, start_x):
    return (intersection_point[0], start_x)  # X軸を合わせる

# 直線の途中で衝突した点をStartのY軸に合わせて描画する
def adjust_point_to_start_y(intersection_point, start_y):
    return (start_y, intersection_point[1])  # Y軸を合わせる

# 移動量を計算する関数
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# 繰り返し処理（3回繰り返し）
new_point = start

for repeat in range(12):  # 最大3回繰り返す
    # StartからGoalまでの直線がぶつかった点を取得
    line_intersections = draw_line_with_intersections(new_point, goal, grid)

    if line_intersections:  # 衝突があった場合
        collision_point = line_intersections[0]
        
        # 衝突点をStartと同じX軸位置にずらす
        new_point_x = adjust_point_to_start_x(collision_point, new_point[1])  # new_point[1]はX軸位置
        new_point_y = adjust_point_to_start_y(collision_point, new_point[0])  # new_point[0]はY軸位置
        
        # 新しい中継点の移動量を計算
        dist_x = distance(new_point, new_point_x)
        dist_y = distance(new_point, new_point_y)

        # 移動量が大きい方を選択して描画
        if dist_x >= dist_y:
            cv2.circle(color_image, (new_point_x[1], new_point_x[0]), 10, (0, 255, 0), -1)  # 中継点 (緑色) 水平
            cv2.line(color_image, (new_point[1], new_point[0]), (new_point_x[1], new_point_x[0]), (0, 255, 0), 2)  # 水平
            new_point = new_point_x  # 次の繰り返しのために新しい中継点をstartにする
        else:
            cv2.circle(color_image, (new_point_y[1], new_point_y[0]), 10, (0, 255, 0), -1)  # 中継点 (緑色) 垂直
            cv2.line(color_image, (new_point[1], new_point[0]), (new_point_y[1], new_point_y[0]), (0, 255, 0), 2)  # 垂直
            new_point = new_point_y  # 次の繰り返しのために新しい中継点をstartにする
        
    else:
        print(f"Iteration {repeat + 1}: No intersection found.")
        break

# 結果の画像をPIL形式に変換
final_image = Image.fromarray(color_image)

# 二値化後の画像を保存
final_image.save('test_output.jpg')

# 画像を表示
final_image.show()
