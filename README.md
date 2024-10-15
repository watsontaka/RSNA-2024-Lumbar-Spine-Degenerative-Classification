# RSNA 2024 Lumbar Spine Degenerative Classification

# コンペティションの概要

各方向から撮影された腰椎のMRI画像を使って腰椎の５つの部位の状態を予測するという内容のコンペティションになります。

Normal/Mild　正常/軽症

Moderate 　　中程度

Severe       重症

コンペティションのURLは以下になります。

[https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification]

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

* Left Subarticular Stenosis

* Right Subarticular Stenosis


## MRIについて

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

対応表

Spinal Canal Stenosis - SaggitalT2/STIR

Neural Foraminal Stenosis - SaggitalT1

Subarticular Stenosis - AxialT2

dicom .dcm

医用画像の拡張子　患者の情報や画像の位置関係等のメタ情報が含まれています。

pydicom

## ポイント

MRI画像のピクセル値が255を超えるものがあったので、正規化(0-255)の処理をしました。

MRIの画像の枚数は各患者ごとに異なります(7枚しかないものもあれば30枚あるものもあります。)。
ニューラルネットワークで学習させるには枚数が異なるので、枚数をチャンネルとしてそのまま流すことはできません。
そこで、該当する画像をニューラルネットワークで抽出することにしました。

SaggitalT2/STIR 1枚

SaggitalT1 4枚

内訳
Left 2枚
Right 2枚

AxialT2 10枚

Neural Foraminal Stenosis(椎間孔狭窄症)は左側と右側があり、1つのフォルダでdicomファイルが管理されています。
MRI画像は左から右へ

## 機械学習モデル

目的変数
5,3の行列で予測されるように正解ラベルを作成しました。

'''
[[1,0,0]
 [1,0,0]
 [1,0,0]
 [0,1,0]
 [0,0,1]]
'''

画像抽出　DenseNet

部位予測　EfficientNet



## 苦労した点・良かったと思う点

* 私自身医学知識がないので、MRIの仕組みや症状を把握するのが大変でした。

* 全て英語で書かれているので読めるところは自分で読んで、読めないところは翻訳したりと内容の理解が大変でした。

* 今回はCode Competitionのため提出時にhidden data(隠しデータ)でスコアを算出するような仕組みなおですが、当初は意味がわからず提出するのに1週間近くかかりました。
* 他の方が提出に成功しているコードを参考にデバックを行ってやっとの思いで提出できました。
* 結果は良くありませんでしたが、自分オリジナルの機械学習モデルを作れたと思います。
