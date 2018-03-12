# 105 FJU ANN Assignment 02

1. cd /path/to/this/folder/

2. 執行前，須先安裝其它套件。
   pip install -r requirement.txt

3. 執行程式
   python main.py [資料型態] {learning rate} {最大迭代數} {容忍值}

    資料型態：
      problem - 使用題目資料（8 + 4）
      extra   - 使用額外資料（1000 + 10）

    參數：
      -l <number> 設定learning rate
      -m <number> 設定最大迭代次數，以免永不收斂
      -t <number> 設定差異容忍值

    範例:
      python main.py problem -l 0.1 -m 1000 -t 0.001
        - 使用題目資料，指定learning rate = 0.1、最大迭代數 = 1000、容忍值 = 0.001

4. 相關程式碼皆在adaline資料夾中

5. txt資料在data資料夾

6. 使用額外測資(extra)時會將訓練結果製作成PNG圖檔，檔案位置在原目錄下
