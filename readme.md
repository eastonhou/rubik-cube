# 生成状态文件
运行launch/state3
# 生成码表
运行launch/code
# 训练状态模型
运行launch/train
# 训练码模型
运行launch/train.code
# 解算魔方
修改solve_best2.py中L189~L194，运行launch/solve

# ui安装部署
pip install glumpy
copy dll to system32: https://github.com/ubawurinna/freetype-windows-binaries/blob/master/release%20dll/win64/freetype.dll
