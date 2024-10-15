# RSNA 2024 Lumbar Spine Degenerative Classification

# コンペティションの概要

脊柱管狭窄症、ヘルニアなどの原因となる腰椎の部位を検出することを目的としたコンペティションです。

MRI画像を使い腰椎のうちの５つの部位を **通常(Normal/Mild)** **中傷(Moderate)** **重症(Severe)** の３種類の程度の確率を予測する機械学習モデルを作ることが目標になります。

例

|    |Normal/Mild|Moderate|Severe|
|:--:|:--:|:--:|:--:|
|第一腰椎|0.95|0.01|0.04|
|第二腰椎|0.86|0.13|0.01|
|第三腰椎|0.33|0.56|0.11|
|第四腰椎|0.07|0.30|0.63|
|第五腰椎|0.40|0.50|0.10|

コンペティションのURLは以下になります。

[https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification]

ディレクトリの構成

# EDA

## 腰椎

以下の通り画像の通りL1からL5まであります。

![腰椎](https://fuelcells.org/wp/wp-content/uploads/2023/11/intervertebral-disc.jpg.webp)

画像:リペアセルクリニック [https://fuelcells.org/topics/27830/]

### 症状について

* **脊柱管狭窄症(Spinal Canal Stenosis)**
脊柱管という背骨の中にある神経の通り道が狭くなり、神経が圧迫される病気です。
![脊柱管狭窄症]()

* **椎間孔狭窄症(Neural Foraminal Stenosis)**
神経の本幹である脊髄（せきずい）から左右に枝分かれする細い神経の通り道が狭くなり圧迫される病気です。
  * **左側椎間孔狭窄症(Left Neural Foraminal Stenosis)**
  * **右側椎間孔狭窄症(Right Neural Foraminal Stenosis)**
![椎間孔狭窄症]()

* **変性すべり症(Subarticular Stenosis)**
  * **左側変性すべり症(Left Subarticular Stenosis)**
  * **右側変性すべり症(Right Subarticular Stenosis)**
![変性すべり症]()


## MRIについて

MRI画像は断面が３種類あります。

![断面](https://ilclinic.or.jp/wp/wp-content/uploads/2022/12/image-18.png)

* **矢状断面(Sagittal)**

![矢状断面](https://ilclinic.or.jp/wp/wp-content/uploads/2022/12/image-14.png)

* **水平断面(Axial)**
  
![水平断面](https://ilclinic.or.jp/wp/wp-content/uploads/2022/12/image-15.png)

* **冠状断面(Coronal)**

![冠状断](https://ilclinic.or.jp/wp/wp-content/uploads/2022/12/image-17.png)

本コンペティションで用意されている画像は**Saggital画像**と**Axial画像**になります。

MRIの信号の種類

T2/STIR

T2

T1

参考URL
[]

csvファイルを見ると以下の関連性があります。

Spinal Canal Stenosis - SaggitalT2/STIR
矢状断面でT2信号脂肪抑制で撮影した画像になります。

Neural Foraminal Stenosis - SaggitalT1
矢状断面でT1信号で撮影した画像になります。

Subarticular Stenosis - AxialT2
水平断面でT2信号で撮影した画像になります。

参考・参考画像:[https://ilclinic.or.jp/column/mri%e7%94%bb%e5%83%8f%e3%81%ae%e8%a6%8b%e6%96%b9]

dicomファイル

医用画像の拡張子で画像データのみならず、患者の情報や画像の位置関係などのメタ情報が含まれています。




## 前処理

グレースケール画像は(224,224,1)となるように加工しました。
MRI画像のピクセル値が255を超えるものがあったので、正規化(0-255)の処理をしました。
$$\dfrac{X-min}{max-min}$$

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
