# RSNA 2024 Lumbar Spine Degenerative Classification

# コンペティションの概要

脊柱管狭窄症、ヘルニア等の変性脊椎症の原因となる腰椎の部位を検出することを目的としたコンペティションです。
MRI画像を使い腰椎のうちの５つの部位を **通常(Normal/Mild)** **中傷(Moderate)** **重症(Severe)** の３種類の程度の確率を予測する機械学習モデルを作ることが目標になります。

例

|       |Normal/Mild|Moderate|Severe|
|:--:|:--:|:--:|:--:|:--:|
|第一腰椎|
|第二腰椎|
|第三腰椎|
|第四腰椎|
|第五腰椎|

コンペティションのURLは以下になります。

[https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification]

ディレクトリの構成

# EDA

## 腰椎

以下の通り画像の通りL1からL5まであります。

![腰椎](https://fuelcells.org/wp/wp-content/uploads/2023/11/intervertebral-disc.jpg.webp)

画像　リペアセルクリニック
[https://fuelcells.org/topics/27830/]

### 症状について

* **脊柱管狭窄症**(Spinal Canal Stenosis)

* **左側椎間孔狭窄症**(Left Neural Foraminal Stenosis )

*  **右側椎間孔狭窄症**(Right Neural Foraminal Stenosis)

* **右側**(Left Subarticular Stenosis)

* **左側**(Right Subarticular Stenosis)


## MRIについて

MRI画像は断面が３種類ある。

* Saggital - 矢状断面
![矢状断面]()

* Axial - 水平断面
![水平断面]()

MRIの信号の種類

T2/STIR

T2

T1

参考URL
[]

なお症状が確認できる

Spinal Canal Stenosis - SaggitalT2/STIR

Neural Foraminal Stenosis - SaggitalT1

Subarticular Stenosis - AxialT2

dicomファイル

医用画像の拡張子で画像データのみならず、患者の情報や画像の位置関係などのメタ情報が含まれています。




## 前処理

MRI画像のピクセル値が255を超えるものがあったので、正規化(0-255)の処理をしました。

輝度を統一するため、すべての画像に

MRIの画像の枚数は各患者ごとに異なります(7枚しかないものもあれば30枚あるものもあります。)。
ニューラルネットワークで学習させるには枚数が異なるので、枚数をチャンネルとしてそのまま流すことはできません。
そこで、該当する画像をニューラルネットワークで抽出することにしました。

* SaggitalT2/STIR画像 1枚

* SaggitalT1画像 4枚
  * Left 2枚
  * Right 2枚

* AxialT2画像 10枚

Neural Foraminal Stenosis(椎間孔狭窄症)は左側と右側があり、1つのフォルダ内で画像が管理されています。
左から右へもしくは右から左へ順番に切っていったのか一見すると見分けがつきません。
そこで、dicomファイルの位置情報を解析したところ右側から順番に切っている場合にはx軸がマイナスに

## 機械学習モデル

ニューラルネットワークで画像抽出 → 抽出した画像を使って各部位がどの程度の症状なのかを確率で予測



**画像抽出用のモデル**

csvファイルで症状が確認できる画像のデータがあるので、それをもとに目的変数を作成しました。
one-hot表現で
|1.dcm|2.dcm|3.dcm|4.dcm|5.dcm|
|:--:|:--:|:--:|:--:|:--:|
|なし|なし|なし|あり|あり|
~~~python
[[1,0]
 [1,0]
 [1,0]
 [0,1]
 [0,1]]
~~~

ニューラルネットワーク

**DenseNet**


**部位予測　EfficientNet**

**目的変数**
one-hot表現を用いて5✖︎3の行列で正解ラベルを作成しました。

例
|L1/L2|L2/L3|L3/L4|L4/L5|L5/S1|
|:--:|:--:|:--:|:--:|:--:|
|Normal/Mild|Normal/Mild|Normal/Mild|Moderate|Severe|
~~~python
[[1,0,0]
 [1,0,0]
 [1,0,0]
 [0,1,0]
 [0,0,1]]
~~~

ニューラルネットワーク

**EfficientNet**



## 最終的なスコア


## 苦労した点・良かったと思う点

* 私自身医学知識がないので、MRIの仕組みや症状を把握するのが大変でした。

* 全て英語で書かれているので翻訳が大変でした。

* 今回はCode Competitionのため提出時にhidden data(隠しデータ)でスコアを算出するような仕組みですが、当初は意味がわからず提出するのに1週間近くかかりました。
* 他の方が提出に成功しているコードを参考にデバックを行ってやっとの思いで提出できました。
* 結果は良くありませんでしたが、オリジナルの機械学習モデルを作れたと思います。
