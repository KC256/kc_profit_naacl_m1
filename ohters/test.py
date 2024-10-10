import pandas as pd
import matplotlib.pyplot as plt
import mplcursors


# 例として適当なDataFrameを作成
data = {'step': [1, 2, 3, 4, 5],
        'ratio': [0.2, 0.4, 0.6, 0.8, 1.0],
        'INPUT_TEXT': ['A', 'B', 'C', 'D', 'E'],
        'USING_MODEL': ['Model1', 'Model2', 'Model1', 'Model2', 'Model1']}

df = pd.DataFrame(data)

# グラフを散布図としてプロット
scatter = plt.scatter(df['step'], df['ratio'], marker='o', color='blue', label='Data Points')

# x軸、y軸のラベルを設定
plt.xlabel('Step')
plt.ylabel('Ratio')

# グラフにタイトルを追加
plt.title('Scatter Plot of Step vs Ratio')

# 凡例を表示
plt.legend()

# mplcursorsを使用してカーソルを合わせたときに表示するテキストを設定
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(
        f"INPUT_TEXT: {df['INPUT_TEXT'][sel.target.index]}, USING_MODEL: {df['USING_MODEL'][sel.target.index]}"
    )
)
print("hoge")
# グラフを表示
plt.show()
