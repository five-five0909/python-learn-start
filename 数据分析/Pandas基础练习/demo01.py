import pandas as pd
import numpy as np
obj1=pd.Series([4,3,-1,2])
obj2=pd.Series([4,3,-1,2],index=['a','b','c','d'])
print(obj1)
print(obj2)
# In [data], [45]
# 创建一个字典，每个键代表一列的名称，值是该列的数据
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

# 使用 pd.DataFrame() 将字典转换为 DataFrame
frame = pd.DataFrame(data)

# 打印 DataFrame。对于较大的 DataFrame，只会显示摘要信息
print(frame)
# In [98] & [99]
# 创建一个 3x3 的 DataFrame
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
print("--- 原始 DataFrame ---\n", frame)

# In [100] & [101]
# --- 只对行索引进行 reindex ---
# 新的行索引中 'b' 是新增的
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
print("\n--- 对行进行 reindex 后 ---\n", frame2)

# In [102] & [103]
# --- 只对列索引进行 reindex ---
# 使用 columns 关键字对列进行重新索引
# 新的列索引中 'Utah' 是新增的
states = ['Texas', 'Utah', 'California']
frame3 = frame.reindex(columns=states)
print("\n--- 对列进行 reindex 后 ---\n", frame3)

# In [110] & [111]
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
print("--- 原始 DataFrame ---\n", data)

# In [112]
# --- 丢弃行 ---
# 默认 axis=0，所以会删除行索引
dropped_rows = data.drop(['Colorado', 'Ohio'])
print("\n--- 丢弃 'Colorado', 'Ohio' 两行后 ---\n", dropped_rows)

# In [113] & [114]
# --- 丢弃列 ---
# 必须指定 axis=1 或 axis='columns'
dropped_cols = data.drop('two', axis=1)
print("\n--- 丢弃 'two' 列后 ---\n", dropped_cols)

dropped_cols2 = data.drop(['two', 'four'], axis='columns')
print("\n--- 丢弃 'two', 'four' 两列后 ---\n", dropped_cols2)