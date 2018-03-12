# 105 FJU ANN Assignment 01

1. cd /path/to/this/folder/

2. 執行前，須先安裝其它套件。
   pip install -r requirement.txt

3. 執行程式
   python main.py problem -l 0.1 -m 1000

   資料類型：
       problem    使用題目資料（8筆訓練資料 + 4筆測試資料）
       extra      使用額外資料（1000筆訓練資料 + 10筆測試資料）

   參數（必須，數值可自訂）：
       -l 0.1     設定learning rate = 0.1
       -m 1000    設定最大迭代數 = 1000 次

4. 相關程式碼皆在preceptron資料夾中

5. txt資料在data資料夾

6. 使用額外測資(extra)時會將訓練結果製作成PNG圖檔，檔案位置在原目錄下
