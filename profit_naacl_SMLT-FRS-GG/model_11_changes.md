# `model_11.py` の変更点

`Actor`クラスの`forward`メソッドにおいて、`text_out`を計算するロジックを以下の通り変更しました。

## 変更前

`text_out_G1`と`text_out_G7`の単純な平均を計算していました。

```python
# text_out_G1 text_out_G7からtext_outを作成
text_out = (text_out_G1 + text_out_G7)/2
```

## 変更後

`text_out_G1`と`text_out_G7`を連結し、線形層に入力して`pre_text_out`を生成します。
次に、その`pre_text_out`と`text_out_G1`を再度連結し、別の線形層を通して最終的な`text_out`を生成するように変更しました。

```python
# text_out_G1 text_out_G7からtext_outを作成
pre_text_out = self.linear_pre_text_out(torch.cat([text_out_G1, text_out_G7], dim=2))
text_out = self.linear_text_out(torch.cat([pre_text_out, text_out_G1], dim=2))
```

この変更により、2つの異なる粒度のテキスト特徴量(`text_out_G1`, `text_out_G7`)から、より複雑な関係性を学習して特徴量を生成することが期待されます。

`forward`処理中の`text_out`の形状は、指定通り`torch.Size([20, 1, 64])`となります（その後の`permute`と`view`操作前）。
