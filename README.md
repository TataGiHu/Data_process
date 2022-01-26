

#### 简介
提取bag中车道线相关的数据，用于中心线生成训练



```
extract_bag.py 
作用: 用于抽取bag中的 /perception/vision/lane, /worldmodel/processed_map, /mla/egopose 三个topic， 并做同步，写到文件中 
用法: 
python extract_bag.py -b  /to/bag/dir/ -s /save/path/

-b: 存有bag的目录 
-s: 保存提取结果的目录 
```

```
generate_traindata.py 
作用: 根据extract_bag 提取出来的数据，生成训练所用数据 
用法: 
python generate_traindata.py -b /to/data/dir/ -s /save/path
-b: 存有用 extract_bag 提取出来的数据的目录
-s: 保存训练数据，按文件进行区分 
```




运行vis
```
sudo apt-get install ffmpeg
sudo apt-get install python3-tk
```







