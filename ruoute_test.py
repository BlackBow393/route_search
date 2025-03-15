import cv2
import numpy as np
import heapq

# 画像の読み込み
image_path = 'test.jpg'  # パスを確認

# 画像の読み込みを確認
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 画像が正常に読み込めているか確認
if image is None:
    raise ValueError(f"画像ファイルが読み込めませんでした: {image_path}")

# サイズ指定（例えば、幅800px、高さ600pxにリサイズ）
resize_width = 800
resize_height = 600
resized_image = cv2.resize(image, (resize_width, resize_height))

# A*アルゴリズムの実装
class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]
        self.start = (self.rows - 1, 0)  # 左下
        self.end = (0, self.cols - 1)  # 右上
        self.open_list = []
        self.closed_list = set()
        self.g_scores = np.inf * np.ones((self.rows, self.cols))
        self.g_scores[self.start] = 0
        self.f_scores = np.inf * np.ones((self.rows, self.cols))
        self.f_scores[self.start] = self.heuristic(self.start)
        self.came_from = {}

    def heuristic(self, point):
        # ヒューリスティック関数（マンハッタン距離）
        return abs(point[0] - self.end[0]) + abs(point[1] - self.end[1])

    def get_neighbors(self, point):
        # 隣接するノード（上下左右）の取得
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in directions:
            neighbor = (point[0] + direction[0], point[1] + direction[1])
            if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                if self.grid[neighbor] == 0:  # 障害物（黒）は通行不可（通行可能な領域は0）
                    neighbors.append(neighbor)
        return neighbors

    def find_path(self):
        heapq.heappush(self.open_list, (self.f_scores[self.start], self.start))

        while self.open_list:
            _, current = heapq.heappop(self.open_list)
            if current == self.end:
                # ゴールに到達した場合、経路を戻す
                path = []
                while current in self.came_from:
                    path.append(current)
                    current = self.came_from[current]
                return path[::-1]  # 逆順にして経路を返す

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue
                tentative_g_score = self.g_scores[current] + 1  # 移動コストは1
                if tentative_g_score < self.g_scores[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_scores[neighbor] = tentative_g_score
                    self.f_scores[neighbor] = self.g_scores[neighbor] + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (self.f_scores[neighbor], neighbor))

        return []  # 経路が見つからなかった場合

# グリッド（リサイズ後の画像）の準備
grid = resized_image

# A*アルゴリズムで最短経路を計算
astar = AStar(grid)
path = astar.find_path()

# 経路を描画
for point in path:
    # 通常、画像座標系では (y, x) の順番なので注意
    cv2.circle(resized_image, (point[1], point[0]), 1, (0, 0, 255), -1)

# 結果の画像を保存
cv2.imwrite('path_image_resized_left_bottom_start.jpg', resized_image)

# 結果を表示
cv2.imshow('Path', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
