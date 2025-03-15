import cv2
import numpy as np
import pandas as pd
import heapq
from PIL import Image
import tkinter as tk
from tkinter import messagebox

# ExcelファイルからStartとGoalの座標を取得する関数
def read_coordinates_from_excel(file_path):
    # Excelファイルを読み込み、シート名とテーブルを指定
    df = pd.read_excel(file_path, sheet_name="通過地点表", header=0)  # header=0 は最初の行を列名として読み込む
    
    # 座標データをリストとして取得
    coordinates = []
    for index, row in df.iterrows():
        coordinates.append((row['Y座標'], row['X座標']))  # (Y, X) 形式で座標を取得

    return coordinates

# 画像を読み込む
image = Image.open('test_source.png')

# グレースケールに変換
gray_image = image.convert('L')

# 画像のサイズを取得
width, height = gray_image.size

# 画像をNumPy配列に変換（OpenCV用）
opencv_image = np.array(gray_image)

# 二値化処理（白い領域を障害物として扱う）
_, binary_image = cv2.threshold(opencv_image, 127, 255, cv2.THRESH_BINARY_INV)  # 黒い部分を白として扱う

# グリッド（画像の各ピクセルが障害物かどうかを示す2D配列）
grid = binary_image

# A*アルゴリズムにおける移動の定義（4方向）
move_directions = [(0, 10), (10, 0), (0, -10), (-10, 0)]  # 10ピクセル単位の移動

# ヒューリスティック関数（マンハッタン距離）
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A*アルゴリズムの実装
def astar(start, goal, grid):
    open_list = []  # 開始ノード
    closed_list = set()  # 探索済みノード
    came_from = {}  # 経路情報
    g_score = {start: 0}  # g(n): 開始ノードから現在のノードまでのコスト
    f_score = {start: heuristic(start, goal)}  # f(n): g(n) + h(n)

    heapq.heappush(open_list, (f_score[start], start))  # 優先度付きキューに開始点を追加

    while open_list:
        # 最小のf_scoreを持つノードを取り出す
        _, current = heapq.heappop(open_list)
        if current == goal:
            # ゴールに到達したら経路を復元
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_list.add(current)

        for direction in move_directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # 隣接ノードがグリッド内かつ障害物でないかをチェック
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 255:  # 障害物（白い領域）があればスキップ
                    continue
            else:
                continue  # 範囲外の座標はスキップ

            if neighbor in closed_list:
                continue

            tentative_g_score = g_score[current] + 10  # 10ピクセルの移動コスト

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # ゴールに到達できない場合

# Excelから通過地点を順番に取得
coordinates = read_coordinates_from_excel('通過地点表.xlsx')

# 最初の画像をベースに描画
color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# すべての経路を順番に描画
for i in range(len(coordinates) - 1):
    start = coordinates[i]
    goal = coordinates[i + 1]
    
    # A*アルゴリズムで最短経路を探索
    path = astar(start, goal, grid)

    # 最短経路が見つかった場合に描画
    if path:
        # 経路を描画
        for (y, x) in path:
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)  # 赤い点で経路を描画

        # 出発点とゴール点を描画
        cv2.circle(color_image, (start[1], start[0]), 10, (0, 255, 0), -1)  # 緑色の点
        cv2.circle(color_image, (goal[1], goal[0]), 10, (0, 255, 0), -1)  # 緑色の点

    else:
        # 最短経路が見つからなかった場合、メッセージボックスで警告
        root = tk.Tk()
        root.withdraw()  # ウィンドウを表示せずにメッセージボックスのみ表示
        messagebox.showerror("エラー", f"最短経路の算出に失敗しました\nStart: {start} -> Goal: {goal}")
        root.destroy()

# 結果の画像をリサイズして表示する
resized_image = cv2.resize(color_image, (1000, 1000))  # 画像を1000x1000にリサイズ（必要に応じて調整）

# 結果の画像を表示
cv2.imshow('All Shortest Paths', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 結果の画像を保存
cv2.imwrite("all_shortest_paths_result.jpg", resized_image)

