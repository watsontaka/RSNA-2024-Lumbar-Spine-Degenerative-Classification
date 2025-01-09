# RSNA 2024 Lumbar Spine Degenerative Classification

# コンペティションの概要

脊柱管狭窄症、ヘルニアなどの原因となっている腰椎の部位を検出することを目的としたコンペティションです。

MRI画像を使い腰椎の５つ(第１腰椎から第５腰椎)の部位を **通常(Normal/Mild)** **中傷(Moderate)** **重症(Severe)** の３種類の確率を予測する機械学習モデルを作ることが目標になります。
以下のような形式でcsvで結果を提出し、スコアを競います。

例:

|    |Normal/Mild|Moderate|Severe|
|:--:|:--:|:--:|:--:|
|spinal_canal_stenosis_L1_L2 |0.95|0.01|0.04|
|spinal_canal_stenosis_L2_L3 |0.86|0.13|0.01|
|spinal_canal_stenosis_L3_L4 |0.33|0.56|0.11|
|spinal_canal_stenosis_L4_L5 |0.07|0.30|0.63|
|spinal_canal_stenosis_L5_S1 |0.40|0.50|0.10|


|    |Normal/Mild|Moderate|Severe|
|:--:|:--:|:--:|:--:|
|L1/L2|0.95|0.01|0.04|
|L2/L3|0.86|0.13|0.01|
|L3/L4|0.33|0.56|0.11|
|L4/L5|0.07|0.30|0.63|
|L5/S1|0.40|0.50|0.10|

コンペティションのURL

[https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification]

# EDA

## 腰椎

腰椎の構造は以下の画像の通りになります。

![腰椎](https://fuelcells.org/wp/wp-content/uploads/2023/11/intervertebral-disc.jpg.webp)

画像:リペアセルクリニック [https://fuelcells.org/topics/27830/]

### 症状について

* **脊柱管狭窄症(Spinal Canal Stenosis)**
脊柱管という背骨の中にある神経の通り道が狭くなり、神経が圧迫される病気です。
![脊柱管狭窄症]()

* **椎間孔狭窄症(Neural Foraminal Stenosis)**
神経の本幹である脊髄から左右に枝分かれする細い神経の通り道(椎間孔)が狭くなり、神経が圧迫される病気です。
症状としては腰部の痛みや手足の痺れが現れます。
  * **左側椎間孔狭窄症(Left Neural Foraminal Stenosis)**
  * **右側椎間孔狭窄症(Right Neural Foraminal Stenosis)**
![椎間孔狭窄症]()

* **変性すべり症(Subarticular Stenosis)**
  * **左側変性すべり症(Left Subarticular Stenosis)**
  * **右側変性すべり症(Right Subarticular Stenosis)**
![変性すべり症]()

本コンペティションでは以下の５つの症状を与えられたMRI画像から**通常(Normal/Mild)** **中傷(Moderate)** **重症(Severe)** の程度を部位ごとにそれぞれ確率で予測することになります。

脊柱管狭窄症(Spinal Canal Stenosis)
左側椎間孔狭窄症(Left Neural Foraminal Stenosis)
右側椎間孔狭窄症(Right Neural Foraminal Stenosis)
左側変性すべり症(Left Subarticular Stenosis)
右側変性すべり症(Right Subarticular Stenosis)

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

**MRIの信号の種類**

|部位|T1|T2|STIR|
|:--:|:--:|:--:|:--:|
|椎体|高信号(白)|中信号(灰色)|低信号(黒)|
|脊柱管|低信号(黒)|高信号(白)|高信号(白)|
|椎間板|中信号(灰色)|中信号(灰色)|高信号(白)|
|脂肪|高信号(白)|中信号(灰色)|低信号(黒)|

参考・参考画像:ILC国際腰痛クリニック [https://ilclinic.or.jp/column/mri%e7%94%bb%e5%83%8f%e3%81%ae%e8%a6%8b%e6%96%b9]


## データセットの概要

* study_id 患者のid
* series_id MRI画像のid　基本各患者ごとに３種類、場合によっては４種類以上あるものもある。
* description 画像の種類 SaggitalT2/STIR画像 SaggitalT1画像 AxialT2画像
* instance_number 症状が現れているMRI画像のナンバー
* condition 症状　**通常(Normal/Mild)** **中傷(Moderate)** **重症(Severe)** の３種類

csvファイルを見ると以下の関連性があることがわかりました。

* 脊柱管狭窄症(Spinal Canal Stenosis)はSaggitalT2/STIR画像から症状を判断してる。

* 椎間孔狭窄症(Neural Foraminal Stenosis)はSaggitalT1画像から症状を判断している。

* Subarticular StenosisはAxialT2画像から判断している。



dicomファイル

医用画像の拡張子で画像データのみならず、患者の情報や画像の位置関係などのメタ情報が含まれています。

参考


## データの前処理

用意されてるMRI画像は高さ、幅がバラバラ
→  224×224の白黒画像に加工しました。

MRI画像のピクセル値が255を超えるものがあった。
→  以下の式を適用して正規化(0-255)の処理をしました。

$$\dfrac{X-min}{max-min}+1e-6*255$$


## 機械学習モデル

**ニューラルネットワークで画像抽出 → 抽出した画像を使って各部位の症状を予測**の二段階の構成を考えました。

1.テストデータのMRI画像を読み取る。

2.**DenceNet**で画像抽出をする。

3.2で抽出した画像をもとに**EfficientNet**で確率を出力する。

4.1~3を繰り返す。

### 画像抽出用のモデル

**目的変数について**

csvファイルに画像のインスタンスナンバーのデータがあり、それをもとにone-hot表現(該当する場合は1,それ以外は0)で目的変数を作成しました。

例：ディレクトリ内の画像が５枚だった場合
|ファイル名|該当しているか|
|:--:|:--:|
|1.dcm|なし|
|2.dcm|なし|
|3.dcm|なし|
|4.dcm|あり|
|5.dcm|あり|

~~~python

# ありの場合
[0,1]
#なしの場合
[1,0]

[[1,0]
 [1,0]
 [1,0]
 [0,1]
 [0,1]]
~~~

**ニューラルネットワーク**

ニューラルネットワークの構造は**DenseNet**を使用しました。
SagittalT2/STIR画像抽出用、SagittalT1画像抽出用、AxialT2画像抽出用それぞれ用意し、症状が現れている画像を以下のとおり抽出することにしました。

* SaggitalT2/STIR画像 1枚
~~~python
# バリデーションデータで約9割5分の抽出に成功しました。
Epoch 20/20
267/267 ━━━━━━━━━━━━━━━━━━━━ 21s 77ms/step - accuracy: 0.9841 - loss: 0.0389 - val_accuracy: 0.9523 - val_loss: 0.1137
~~~

* SaggitalT1画像 合計4枚
  * 左側2枚
  * 右側2枚
~~~python
# バリデーションデータで約9割の抽出に成功しました。
Epoch 20/20
267/267 ━━━━━━━━━━━━━━━━━━━━ 20s 76ms/step - accuracy: 0.9713 - loss: 0.0764 - val_accuracy: 0.8954 - val_loss: 0.3619
~~~

* AxialT2画像 合計10枚
  * L1/L2の部位 2枚
  * L2/L3の部位 2枚
  * L3/L4の部位 2枚
  * L4/L5の部位 2枚
  * L5/S1の部位 2枚
~~~python
# バリデーションデータで約8割5分の抽出に成功しました。
Epoch 20/20
280/280 ━━━━━━━━━━━━━━━━━━━━ 21s 76ms/step - accuracy: 0.9780 - loss: 0.0603 - val_accuracy: 0.8661 - val_loss: 0.4534
~~~

工夫した点
椎間孔狭窄症(Neural Foraminal Stenosis)の症状が読み取れる画像は1つのフォルダ内で管理されていて
右左それぞれの状態を予測する必要があるため、左側画像と右側画像に分けて抽出する必要がありました。

dicomファイルのメタ情報を調べたところ、断面の位置情報を表してるデータを発見し以下のことが判明しました。

* 右側から順番に切っている場合はx軸の値が小さい

* 左側から順番に切っている場合はx軸の値が大きい

上記を踏まえた上で以下のアプローチ方法を考えました。

1.フォルダ内の画像全ての位置情報を抽出する。

2.抽出した位置情報の平均を計算

3.平均より小さければ右側、大きければ左側に分ける。


### 部位予測用のモデル

**目的変数**
one-hot表現を用いて5✖︎3の行列で正解ラベルを作成しました。

例
|L1/L2|L2/L3|L3/L4|L4/L5|L5/S1|
|:--:|:--:|:--:|:--:|:--:|
|Normal/Mild|Normal/Mild|Normal/Mild|Moderate|Severe|


~~~python

# Normal/Mildの場合
[1,0,0]
# Moderateの場合
[0,1,0]
# Severeの場合
[0,0,1]


[[1,0,0]
 [1,0,0]
 [1,0,0]
 [0,1,0]
 [0,0,1]]
~~~

**ニューラルネットワーク**

**EfficientNet**

ニューラルネットワークは**EfficientNet**の構造を使い、脊柱管狭窄症(Spinal Canal Stenosis)予測用、椎間孔狭窄症(Neural Foraminal Stenosis)予測用、変性すべり症(Subarticular Stenosis)予測用の３種類用意しました。

* 脊柱管狭窄症(Spinal Canal Stenosis)
~~~python
Epoch 30/30
25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.9191 - loss: 0.2037 - val_accuracy: 0.7600 - val_loss: 0.6664
~~~
* 椎間孔狭窄症(Neural Foraminal Stenosis)
~~~python
Epoch 30/30
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 88ms/step - Left_accuracy: 0.8927 - Right_accuracy: 0.8838 - loss: 0.6238 - val_Left_accuracy: 0.7300 - val_Right_accuracy: 0.7240 - val_loss: 1.4949
~~~
* 変性すべり症(Subarticular Stenosis)
~~~python
Epoch 30/30
25/25 ━━━━━━━━━━━━━━━━━━━━ 3s 122ms/step - Left_accuracy: 0.8795 - Right_accuracy: 0.9219 - loss: 0.4973 - val_Left_accuracy: 0.7620 - val_Right_accuracy: 0.7320 - val_loss: 1.6870
~~~
## 最終的なスコア


## 苦労した点・良かったと思う点

* 私自身医学知識がないので、MRIの仕組みや症状を把握するのが特に苦労しました。

* 全て英語で書かれているので翻訳が大変でした。

* 本コンペはCode Competitionであるため、提出時にhidden data(隠しデータ)を使ってスコアを算出するような仕組みですが、当初はその意味がわからず提出するのに1週間近くかかりました。
* 
* 他の方が提出に成功しているコードを参考にデバックを行い、どこが間違っているのかを突き止められました。
* 
* 結果は良くありませんでしたが、オリジナルの機械学習モデルを作れたと思います。
