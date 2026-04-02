### **Pandas 初学者完全指南**

欢迎来到 Pandas 的学习之旅！Pandas 是 Python 数据分析领域的基石。它提供了强大、灵活且易于使用的数据结构，让数据清洗和分析工作变得前所未有的高效。

Pandas 的设计深受另一个库 **NumPy** 的影响，并大量使用了 NumPy 的功能。可以简单理解为：NumPy 是处理纯数字“数组”的专家，而 Pandas 则是处理带有标签的、更复杂的“表格”数据的王者。

**约定：**
在接下来的所有代码中，我们将遵循一个通用约定，将 pandas 导入并简写为 `pd`。

```python
# 导入 pandas 库，并使用 "pd" 作为它的简称（别名），这是数据科学领域的标准做法
import pandas as pd
# 导入 numpy 库，并使用 "np" 作为它的简称，我们经常需要它来生成数据
import numpy as np
```

---

### **5.1 Pandas 的数据结构介绍**

要掌握 Pandas，首先必须熟悉它的两个核心数据结构：`Series` 和 `DataFrame`。

#### **Series (序列)**

`Series` 是一种类似于一维数组的对象，是构成 `DataFrame` 的基础。你可以把它想象成 Excel 表格中的**一列**数据。它由两部分组成：

1.  **值 (Values)**：一组数据。
2.  **索引 (Index)**：与每个数据点相关联的标签。

##### **1. 创建一个简单的 `Series`**

我们可以直接从一个 Python 列表中创建最基础的 `Series`。

```python
# In [11] & [12]
# 创建一个 Series 对象。因为我们没有指定索引，pandas 会自动创建一个从 0 开始的整数索引。
obj = pd.Series([4, 7, -5, 3])

# 打印这个 Series
print(obj)
```

**输出结果及解读:**
```
0    4
1    7
2   -5
3    3
dtype: int64
```
*   **左侧 (0, 1, 2, 3)** 是自动生成的**索引 (Index)**。
*   **右侧 (4, 7, -5, 3)** 是我们传入的**值 (Values)**。
*   **`dtype: int64`** 说明这些值的数据类型是64位整数。

##### **2. 查看 `Series` 的值和索引**

我们可以通过 `.values` 和 `.index` 属性单独访问 `Series` 的值和索引对象。

```python
# In [13] & [14]
# 使用 .values 属性，可以获取 Series 中的所有值，返回的是一个 NumPy 数组
print("Series 的值:", obj.values)

# 使用 .index 属性，可以获取 Series 的索引对象
print("Series 的索引:", obj.index)
```
**输出结果:**
```
Series 的值: [ 4  7 -5  3]
Series 的索引: RangeIndex(start=0, stop=4, step=1)
```

##### **3. 创建带有自定义索引的 `Series`**

为数据打上有意义的标签，会让数据处理和分析变得更加直观。

```python
# In [15] & [16]
# 在创建 Series 时，通过 index 参数传入一个列表，为每个值指定一个标签
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

# 打印新的 Series
print(obj2)
```
**输出结果:**
```
d    4
b    7
a   -5
c    3
dtype: int64
```

##### **4. 通过索引选取和修改数据**

有了自定义索引，我们可以像操作 Python 字典一样方便地选取和修改数据。

```python
# In [18], [19], [20]
# --- 选取单个值 ---
# 通过索引标签 'a' 获取对应的值
print("索引 'a' 的值:", obj2['a'])  # 输出: -5

# --- 修改单个值 ---
# 通过索引 'd' 定位到数据，并将其值修改为 6
obj2['d'] = 6
print("\n修改索引 'd' 后的 Series:\n", obj2)

# --- 选取多个值 ---
# 传入一个包含多个索引标签的列表，可以一次性获取这些标签对应的值，返回一个新的 Series
print("\n选取 'c', 'a', 'd' 三个值:\n", obj2[['c', 'a', 'd']])
```

##### **5. `Series` 的向量化运算**

`Series` 的强大之处在于，它支持**向量化运算**，这意味着我们可以对整个序列进行数学运算、逻辑判断等，而无需编写循环。这个特性继承自 NumPy。

```python
# In [21]
# --- 布尔过滤 ---
# 选取 obj2 中所有值大于 0 的元素
print("obj2 中大于 0 的元素:\n", obj2[obj2 > 0])

# In [22]
# --- 标量乘法 ---
# 将 obj2 中的每个元素都乘以 2
print("\nobj2 中每个元素乘以 2:\n", obj2 * 2)

# In [23]
# --- 应用数学函数 ---
# 使用 NumPy 的 exp 函数计算每个元素的自然指数 e^x
print("\n对 obj2 中每个元素应用 np.exp 函数:\n", np.exp(obj2))
```

##### **6. 将 `Series` 视作字典**

`Series` 的“索引-值”结构与字典的“键-值”非常相似，因此我们可以用类似字典的方式操作它。

```python
# In [24] & [25]
# 使用 'in' 关键字检查一个索引是否存在于 Series 中
print("'b' in obj2:", 'b' in obj2)  # 输出: True
print("'e' in obj2:", 'e' in obj2)  # 输出: False
```

##### **7. 从字典创建 `Series`**

我们也可以直接从一个 Python 字典来创建 `Series`，字典的键将自动成为 `Series` 的索引。

```python
# In [26] & [27]
# 创建一个存储州人口的字典
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

# 将字典转换为 Series
obj3 = pd.Series(sdata)
print(obj3)
```
**输出结果:**
```
Ohio      35000
Texas     71000
Oregon    16000
Utah       5000
dtype: int64
```

##### **8. 处理缺失数据 (`NaN`)**

在数据处理中，经常会遇到数据缺失的情况。Pandas 使用 `NaN` (Not a Number) 来表示缺失值。

当我们根据一个指定的索引列表来创建 `Series` 时，如果某些索引在原始数据中找不到对应的值，就会产生 `NaN`。

```python
# In [29], [30], [31]
# 原始数据字典
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
# 我们希望的索引顺序和范围
states = ['California', 'Ohio', 'Oregon', 'Texas']

# 基于 sdata 字典创建 Series，但使用我们新指定的 states 列表作为索引
obj4 = pd.Series(sdata, index=states)
print(obj4)
```
**输出结果解读:**
```
California        NaN  # 'California' 在 sdata 中不存在，所以值为 NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64          # 因为 NaN 的存在，数据类型自动变为浮点型
```
*   **注意:** `Utah` 因为不在我们指定的 `states` 索引列表中，所以它被丢弃了。

##### **9. 检测缺失数据**

Pandas 提供了 `isnull()` 和 `notnull()` 两个函数来方便地检测缺失数据。

```python
# In [32] & [33]
# 使用 pd.isnull() 检查 obj4 中的缺失值，返回一个布尔值的 Series
print("检查 obj4 中的缺失值 (isnull):\n", pd.isnull(obj4))

# 使用 pd.notnull() 检查 obj4 中的非缺失值
print("\n检查 obj4 中的非缺失值 (notnull):\n", pd.notnull(obj4))

# 也可以作为 Series 自身的方法来调用
# print("\n使用方法调用 isnull:\n", obj4.isnull())
```

##### **10. `Series` 的自动对齐**

Pandas 最核心的功能之一就是**数据对齐**。当对两个 `Series` 进行算术运算时，Pandas 会自动按索引对齐数据。如果某个索引只存在于其中一个 `Series`，则结果中该索引对应的值为 `NaN`。

```python
# In [35], [36], [37]
print("--- 原始数据 obj3 ---\n", obj3)
print("\n--- 原始数据 obj4 ---\n", obj4)

# 将 obj3 和 obj4 相加
# Pandas 会找到两个 Series 中共有的索引 ('Ohio', 'Oregon', 'Texas') 并将它们的值相加。
# 对于只在一方存在的索引 ('California', 'Utah')，结果会是 NaN。
result = obj3 + obj4
print("\n--- obj3 + obj4 的结果 ---\n", result)
```

##### **11. `Series` 的 `name` 属性**

`Series` 对象本身和它的索引都可以有一个 `name` 属性，这在处理多个数据对象时非常有用，能让数据更具可读性。

```python
# In [38], [39], [40]
# 为 obj4 这个 Series 本身命名
obj4.name = 'population'
# 为 obj4 的索引命名
obj4.index.name = 'state'

print(obj4)
```
**输出结果:**
```
state
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
Name: population, dtype: float64
```

##### **12. 修改 `Series` 的索引**

你可以直接对 `Series` 的 `index` 属性进行赋值，来替换掉原有的索引。

```python
# In [41], [42], [43]
# 原始的 obj
print("--- 原始 Series ---\n", obj)

# 直接为 index 属性赋一个新的列表，列表长度必须与 Series 长度一致
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print("\n--- 修改索引后 ---\n", obj)
```

---

#### **DataFrame (数据帧)**

`DataFrame` 是一个二维的、表格型的数据结构，是 Pandas 中最核心、使用最广泛的对象。你可以把它想象成一个功能更强大的 Excel 电子表格或 SQL 数据表。

**`DataFrame` 的特点：**
*   含有一组有序的**列**。
*   每列可以是不同的数据类型（数值、字符串、布尔值等）。
*   同时拥有**行索引 (index)** 和**列索引 (columns)**。
*   可以被看作是一个由 `Series` 组成的字典，这些 `Series` 共享同一个行索引。

##### **1. 创建 `DataFrame`**

最常用的创建方式是传入一个由等长列表或 NumPy 数组组成的字典。

```python
# In [data], [45]
# 创建一个字典，每个键代表一列的名称，值是该列的数据
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

# 使用 pd.DataFrame() 将字典转换为 DataFrame
frame = pd.DataFrame(data)

# 打印 DataFrame。对于较大的 DataFrame，只会显示摘要信息
print(frame)
```
**输出结果:**
```
    state  year  pop
0    Ohio  2000  1.5
1    Ohio  2001  1.7
2    Ohio  2002  3.6
3  Nevada  2001  2.4
4  Nevada  2002  2.9
5  Nevada  2003  3.2
```

##### **2. 查看头部数据**

对于大型 `DataFrame`，我们通常使用 `.head()` 方法来预览前几行数据，对数据有个初步的了解。

```python
# In [46]
# 默认显示前 5 行
print(frame.head())

# 也可以指定行数
# print(frame.head(3))
```

##### **3. 指定列的顺序**

在创建 `DataFrame` 时，Pandas 会自动按字典键的顺序排列列。如果你希望指定列的顺序，可以通过 `columns` 参数实现。

```python
# In [47]
# 即使字典中 'state' 在前，我们也可以通过 columns 参数指定 'year' 列排在最前面
frame_ordered = pd.DataFrame(data, columns=['year', 'state', 'pop'])
print(frame_ordered)
```

##### **4. 处理不存在的列**

如果 `columns` 参数中指定的列在原始数据字典中不存在，那么在结果中这一列的所有值都会是 `NaN`。

```python
# In [48] & [49]
# 我们增加了 'debt' 这一列，它在原始 data 字典中不存在
# 同时我们还通过 index 参数自定义了行索引
frame2 = pd.DataFrame(data,
                      columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four', 'five', 'six'])
print(frame2)
```
**输出结果:**
```
       year   state  pop debt
one    2000    Ohio  1.5  NaN
two    2001    Ohio  1.7  NaN
three  2002    Ohio  3.6  NaN
four   2001  Nevada  2.4  NaN
five   2002  Nevada  2.9  NaN
six    2003  Nevada  3.2  NaN
```

##### **5. 获取列**

获取 `DataFrame` 的列非常简单，有两种方式。获取到的一列是一个 `Series` 对象。

```python
# In [51] & [52]
# --- 方式一：字典式 [ ] 语法 (推荐) ---
# 这种方式适用于所有列名，即使列名中包含空格或特殊字符
state_col = frame2['state']
print("--- 使用 ['state'] 获取列 ---\n", state_col)
print("类型是:", type(state_col))

# --- 方式二：属性 . 语法 ---
# 这种方式更简洁，但只在列名是合法的 Python 变量名时才有效
year_col = frame2.year
print("\n--- 使用 .year 获取列 ---\n", year_col)
```

##### **6. 获取行**

获取行需要使用 `.loc`（基于标签索引）或 `.iloc`（基于整数位置索引）属性。

```python
# In [53]
# --- 使用 .loc 按标签获取行 ---
# 获取行索引为 'three' 的那一行数据
row_three = frame2.loc['three']
print("--- 使用 .loc['three'] 获取行 ---\n", row_three)

# --- 补充示例：使用 .iloc 按位置获取行 ---
# 获取第三行的数据 (整数位置为 2)
row_pos_2 = frame2.iloc[2]
print("\n--- 使用 .iloc[2] 获取行 ---\n", row_pos_2)
```

##### **7. 修改列**

可以直接对列进行赋值来修改或创建新列。

```python
# In [54] & [56]
# --- 赋一个单一值 ---
# 将 'debt' 列的所有值都设为 16.5
frame2['debt'] = 16.5
print("--- 'debt' 列赋标量值后 ---\n", frame2)

# --- 赋一个数组/列表 ---
# 数组的长度必须和 DataFrame 的行数一致
frame2['debt'] = np.arange(6.)
print("\n--- 'debt' 列赋数组后 ---\n", frame2)

# --- 赋一个 Series ---
# 当赋的值是一个 Series 时，会根据索引进行对齐，找不到的索引位置会填充 NaN
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
print("\n--- 'debt' 列赋 Series 后 ---\n", frame2)
```

##### **8. 创建新列和删除列**

为不存在的列名赋值，就会创建出一个新列。`del` 关键字可以用来删除列。

```python
# In [61] & [63]
# --- 创建新列 ---
# 根据 'state' 列的值创建一个布尔类型的'eastern'列
frame2['eastern'] = (frame2['state'] == 'Ohio')
print("--- 创建 'eastern' 列后 ---\n", frame2)

# --- 删除列 ---
del frame2['eastern']
print("\n--- 删除 'eastern' 列后 ---\n", frame2)
print("\n剩下的列名:", frame2.columns)
```

##### **9. 从嵌套字典创建 `DataFrame`**

如果传入一个嵌套字典，外层字典的键会被当作**列索引**，内层字典的键会被当作**行索引**。

```python
# In [65] & [66]
# 外层键 'Nevada', 'Ohio' 是列
# 内层键 2000, 2001, 2002 是行
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = pd.DataFrame(pop)
print(frame3)
```
**输出结果:**
```
      Nevada  Ohio
2001     2.4   1.7
2002     2.9   3.6
2000     NaN   1.5
```
*   `Nevada` 在 2000 年没有数据，所以是 `NaN`。
*   行索引是所有内层键的并集，并自动排序。

##### **10. `DataFrame` 的转置**

使用 `.T` 属性可以像 NumPy 一样对 `DataFrame` 进行转置（行列互换）。

```python
# In [68]
# 对 frame3 进行转置
print(frame3.T)
```

##### **11. `DataFrame` 的 `values` 属性**

与 `Series` 类似，`.values` 属性会以二维 `ndarray` 的形式返回 `DataFrame` 中的数据。

```python
# In [74]
print(frame3.values)
```
**输出结果:**
```
[[2.4 1.7]
 [2.9 3.6]
 [nan 1.5]]
```

---

#### **索引对象 (Index Objects)**

`Index` 对象负责管理轴标签和其他元数据（比如轴名称）。当你创建一个 `Series` 或 `DataFrame` 时，你所提供的任何数组或标签序列都会在内部被转换为一个 `Index` 对象。

##### **1. 索引对象是不可变的**

`Index` 对象的一个重要特性是**不可变 (immutable)**。这意味着你不能像修改列表元素那样去修改一个 `Index` 对象中的某个标签。

```python
# In [76], [77]
obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print("索引对象:", index)

# 尝试修改索引的第二个元素，这会引发一个 TypeError
try:
    index[1] = 'd'
except TypeError as e:
    print("\n错误:", e)
```
**输出结果:**
```
索引对象: Index(['a', 'b', 'c'], dtype='object')

错误: Index does not support mutable operations
```
这种不可变性使得 `Index` 对象可以在多个数据结构之间被安全地共享。

##### **2. 索引对象的方法和属性**

`Index` 对象的行为也像一个固定大小的集合，并拥有很多方法和属性。

**表5-2: `Index` 的主要方法和属性**

| 方法/属性      | 说明                                                       |
| :------------- | :--------------------------------------------------------- |
| `append`       | 连接另一个 `Index` 对象，产生一个新的 `Index`              |
| `difference`   | 计算差集，并得到一个新的 `Index`                           |
| `intersection` | 计算交集                                                   |
| `union`        | 计算并集                                                   |
| `isin`         | 计算一个布尔数组，指示各值是否都包含在参数集合中           |
| `delete`       | 删除指定位置 `i` 的元素，并得到新的 `Index`                |
| `drop`         | 删除传入的值，并得到新的 `Index`                           |
| `insert`       | 将元素插入到指定位置 `i`，并得到新的 `Index`               |
| `is_monotonic` | 当各元素均大于等于前一个元素时，返回 `True`                |
| `is_unique`    | 当 `Index` 没有重复值时，返回 `True`                       |
| `unique`       | 计算 `Index` 中唯一值的数组                                |

```python
# In [87] & [88]
# 检查一个标签是否存在于列索引中
print("'Ohio' in frame3.columns:", 'Ohio' in frame3.columns) # True

# 检查一个标签是否存在于行索引中
print("2003 in frame3.index:", 2003 in frame3.index) # False
```

##### **3. 允许重复的索引**

与 Python 的集合不同，Pandas 的 `Index` 可以包含重复的标签。

```python
# In [89]
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
print("允许重复的索引:", dup_labels)

# 当根据重复标签进行选择时，会选取所有匹配的结果
# (这个内容在后面的 "带有重复标签的轴索引" 章节会详细讲解)
```

---

### **5.2 基本功能**

本章将介绍操作 `Series` 和 `DataFrame` 中数据的核心手段。掌握这些功能是进行数据分析和处理的基础。

#### **重新索引 (Reindexing)**

重新索引 (`reindex`) 是 Pandas 对象的一个非常重要的方法。它的作用是**创建一个新对象**，其数据**符合新的索引**。

##### **1. 对 `Series` 进行重新索引**

`reindex` 会根据新索引对数据进行重排。如果某个索引值当前不存在，就会引入缺失值 `NaN`。

```python
# In [91] & [92]
# 创建一个 Series
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
print("--- 原始 Series ---\n", obj)

# In [93] & [94]
# 使用 reindex 创建一个新对象，并为其指定一个新的索引
# 新索引的顺序是 'a', 'b', 'c', 'd', 'e'
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
print("\n--- 重新索引后 ---\n", obj2)
```
**输出结果解读:**
```
--- 原始 Series ---
 d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64

--- 重新索引后 ---
 a   -5.3  # 'a' 的值被保留
b    7.2  # 'b' 的值被保留
c    3.6  # 'c' 的值被保留
d    4.5  # 'd' 的值被保留
e    NaN  # 'e' 在原始索引中不存在，所以值为 NaN
dtype: float64
```

##### **2. 重新索引时的插值处理**

对于时间序列等有序数据，我们可能希望在重新索引时对缺失值进行“填充”或“插值”。`method` 选项可以实现这个功能。

*   `method='ffill'` 或 `method='pad'`：**前向填充** (Forward Fill)，用前一个有效值来填充 `NaN`。
*   `method='bfill'` 或 `method='backfill'`：**后向填充** (Backward Fill)，用后一个有效值来填充 `NaN`。

```python
# In [95], [96], [97]
# 创建一个索引不连续的有序 Series
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
print("--- 原始 Series ---\n", obj3)

# 对 0-5 的完整索引进行重新索引，并使用前向填充
# 索引 1 的值会使用索引 0 的值 'blue'
# 索引 3 的值会使用索引 2 的值 'purple'
# 索引 5 的值会使用索引 4 的值 'yellow'
obj3_ffill = obj3.reindex(range(6), method='ffill')
print("\n--- 使用 ffill 重新索引后 ---\n", obj3_ffill)
```

##### **3. 对 `DataFrame` 进行重新索引**

`reindex` 在 `DataFrame` 上可以同时修改行索引和列索引。

```python
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
```

**`reindex` 函数的参数 (表5-3)**

| 参数         | 说明                                                                                              |
| :----------- | :------------------------------------------------------------------------------------------------ |
| `index`      | 用作索引的新序列。                                                                                |
| `method`     | 插值（填充）方式，如 `'ffill'`, `'bfill'`。                                                       |
| `fill_value` | 在重新索引的过程中，需要引入缺失值时使用的替代值，而不是 `NaN`。                                    |
| `limit`      | 在前向或后向填充时，允许连续填充的最大数量。                                                      |
| `level`      | 在多重索引（MultiIndex）中，指定在哪个级别上进行匹配。                                            |
| `copy`       | 默认为 `True`，总是复制数据；如果为 `False`，且新旧索引相同时，则不复制。                         |

---

#### **丢弃指定轴上的项 (Dropping)**

`drop` 方法会返回一个在指定轴上删除了指定值的新对象。

##### **1. 对 `Series` 使用 `drop`**

```python
# In [105], [106], [107]
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
print("--- 原始 Series ---\n", obj)

# 丢弃索引 'c'
new_obj = obj.drop('c')
print("\n--- 丢弃 'c' 之后 ---\n", new_obj)

# 也可以一次性丢弃多个索引
new_obj2 = obj.drop(['d', 'c'])
print("\n--- 丢弃 'd' 和 'c' 之后 ---\n", new_obj2)
```

##### **2. 对 `DataFrame` 使用 `drop`**

在 `DataFrame` 中，`drop` 可以删除任意轴上的索引值。默认情况下，它删除的是**行** (axis=0)。

```python
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
```

##### **3. `inplace` 参数**

许多函数（如 `drop`）都有一个 `inplace` 参数。如果设置为 `True`，它会**就地修改**原对象，而**不会返回新对象**。

```python
# In [115] & [116]
print("--- drop 前的 obj ---\n", obj)

# 使用 inplace=True 直接在 obj 上进行修改
obj.drop('c', inplace=True)

print("\n--- 使用 inplace=True drop 后的 obj ---\n", obj)
```
> **⚠️ 注意：** 小心使用 `inplace=True`，因为它会销毁所有被删除的数据，且无法撤销。通常更推荐将结果赋给一个新变量。

---

#### **索引、选取和过滤**

`Series` 和 `DataFrame` 的索引操作是其核心功能。

##### **1. `Series` 的索引**

`Series` 的索引工作方式类似于 NumPy 数组，但它的索引值不只是整数。

```python
# In [117] & [118]
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print("--- 原始 Series ---\n", obj)

# --- 标签索引 ---
print("\nobj['b']:", obj['b'])         # 与字典类似

# --- 位置索引 ---
print("obj[1]:", obj[1])           # 与列表/数组类似

# --- 切片 ---
print("\nobj[2:4]:\n", obj[2:4])     # 按位置切片，不包含结束位置

# --- 标签列表 ---
print("\nobj[['b', 'a', 'd']]:\n", obj[['b', 'a', 'd']]) # 按指定顺序选取

# --- 布尔索引 ---
print("\nobj[obj < 2]:\n", obj[obj < 2]) # 选取所有值小于2的元素
```

**一个特殊的切片规则：**
当使用**标签**进行切片时，其末端是**包含的**！

```python
# In [125]
# 'b' 到 'c' 的切片，包含了 'b' 和 'c'
print("\n使用标签切片 obj['b':'c']:\n", obj['b':'c'])
```

##### **2. `DataFrame` 的索引**

`DataFrame` 的索引主要是为了获取一个或多个**列**。

```python
# In [128] & [129]
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
print("--- 原始 DataFrame ---\n", data)

# In [130] & [131]
# --- 获取单列 ---
print("\ndata['two']:\n", data['two'])

# --- 获取多列 ---
print("\ndata[['three', 'one']]:\n", data[['three', 'one']])
```

`DataFrame` 的 `[]` 运算符也有一些特殊情况可以用于**行**的选择：

*   **切片**：`data[:2]` 会选取前两行。
*   **布尔数组**：`data[data['three'] > 5]` 会选取 `three` 列值大于5的所有行。

```python
# In [132] & [133]
# --- 行切片 ---
print("\ndata[:2] (前两行):\n", data[:2])

# --- 布尔行过滤 ---
print("\n'three' 列 > 5 的所有行:\n", data[data['three'] > 5])
```

##### **3. 使用 `loc` 和 `iloc` 进行更精确的选取**

由于 `[]` 运算符的功能在行和列之间有时会产生歧义，Pandas 提供了两个专门的索引属性，让数据选取变得**清晰、明确、无歧义**：

*   `.loc`：基于**标签 (label)** 的索引。
*   `.iloc`：基于**整数位置 (integer position)** 的索引。

**使用 `loc` (按标签)**
```python
# In [137]
# 选取 'Colorado' 行，以及 'two' 和 'three' 这两列
selection1 = data.loc['Colorado', ['two', 'three']]
print("--- 使用 loc 选取行和列 ---\n", selection1)

# 补充示例：选取多行多列
selection2 = data.loc[['Colorado', 'Utah'], ['one', 'four']]
print("\n--- 使用 loc 选取多行多列 ---\n", selection2)

# 补充示例：使用 loc 进行标签切片 (包含末端)
selection3 = data.loc['Colorado':'New York', 'one':'three']
print("\n--- 使用 loc 进行标签切片 ---\n", selection3)
```

**使用 `iloc` (按位置)**
```python
# In [138]
# 选取第3行(位置2)，以及第4, 1, 2列(位置3, 0, 1)
selection4 = data.iloc[2, [3, 0, 1]]
print("\n--- 使用 iloc 选取行和列 ---\n", selection4)

# In [140]
# 选取第2, 3行(位置1, 2)，以及第4, 1, 2列(位置3, 0, 1)
selection5 = data.iloc[[1, 2], [3, 0, 1]]
print("\n--- 使用 iloc 选取多行多列 ---\n", selection5)

# In [141] (此示例在PDF中有些混用，这里澄清一下)
# loc 进行标签切片
print("\n--- loc 标签切片 ---\n", data.loc[:'Utah', 'two'])

# iloc 进行位置切片 (不包含末端)
print("\n--- iloc 位置切片 ---\n", data.iloc[:3, 1]) # 选取前3行，第2列
```
> **强烈建议：** 当你需要同时选取行和列时，优先使用 `.loc` 和 `.iloc`。这能让你的代码意图更清晰，并避免很多潜在的错误。

**`DataFrame` 索引选项总结 (表5-4)**

| 类型                 | 说明                                                                     |
| :------------------- | :----------------------------------------------------------------------- |
| `df[val]`            | 选取单列或多列。特殊情况下可用于布尔数组过滤行、切片行。                 |
| `df.loc[val]`        | 通过**标签**选取单个或多个行。                                           |
| `df.loc[:, val]`     | 通过**标签**选取单个或多个列。                                           |
| `df.loc[val1, val2]` | 通过**标签**同时选取行和列。                                             |
| `df.iloc[where]`     | 通过**整数位置**选取单个或多个行。                                       |
| `df.iloc[:, where]`  | 通过**整数位置**选取单个或多个列。                                       |
| `df.iloc[where_i, where_j]` | 通过**整数位置**同时选取行和列。                                     |
| `df.at[label_i, label_j]` | 按标签快速选取单个标量值 (比 `loc` 更快)。                             |
| `df.iat[i, j]`       | 按整数位置快速选取单个标量值 (比 `iloc` 更快)。                           |

---

### **5.2 基本功能 (续)**

#### **整数索引的特殊说明**

处理带有整数索引的 Pandas 对象时，初学者可能会感到困惑，因为它与 Python 内置的列表和元组索引语法不同。

```python
# In [144] & [145]
# 创建一个带有整数索引的 Series
ser = pd.Series(np.arange(3.))
print("--- 整数索引的 Series ---\n", ser)

# 此时 ser[-1] 会引发 KeyError，因为它试图寻找标签为 -1 的索引，而不是最后一个元素
try:
    ser[-1]
except KeyError as e:
    print("\n尝试 ser[-1] 引发错误:", e)

# 对于非整数索引，则不会产生歧yi
ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
print("\n--- 非整数索引的 Series 的最后一个元素 ---\n", ser2[-1])
```
为了避免这种混淆，**强烈建议**在处理整数索引时，总是明确使用 `.loc`（按标签）或 `.iloc`（按位置）。

```python
# In [148] & [149]
# ser 的索引是整数 0, 1, 2
# loc[:1] 表示选取标签从开始到 1 (包含1) 的所有行
print("\n--- 使用 .loc[:1] ---\n", ser.loc[:1])

# iloc[:1] 表示选取位置从开始到 1 (不包含1) 的所有行，即第0行
print("\n--- 使用 .iloc[:1] ---\n", ser.iloc[:1])
```

---

#### **算术运算和数据对齐**

Pandas 最重要的功能之一，就是它在进行算术运算时能够**自动对齐不同索引的对象**。

##### **1. `Series` 的对齐**

当将两个 `Series` 相加时，如果存在不同的索引对，则结果的索引就是这两个 `Series` 索引的**并集**。对于只在一方存在的索引，结果会是 `NaN`（缺失值）。

```python
# In [150], [151], [152], [153]
# 创建两个索引不完全相同的 Series
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4.0, 3.1], index=['a', 'c', 'e', 'f', 'g'])
print("--- s1 ---\n", s1)
print("\n--- s2 ---\n", s2)

# In [154]
# 将它们相加
result = s1 + s2
print("\n--- s1 + s2 ---\n", result)
```
**输出结果解读:**
```
--- s1 + s2 ---
 a    5.2  # 7.3 + (-2.1)
c    1.1  # (-2.5) + 3.6
d    NaN  # d 只在 s1 中存在
e    0.0  # 1.5 + (-1.5)
f    NaN  # f 只在 s2 中存在
g    NaN  # g 只在 s2 中存在
dtype: float64
```
> 这就像在数据库中进行自动的**外连接 (OUTER JOIN)**。

##### **2. `DataFrame` 的对齐**

对于 `DataFrame`，对齐操作会同时发生在**行和列**上。

```python
# In [155] - [159]
# 创建两个行、列索引都不完全相同的 DataFrame
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)),
                   columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])

df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                   columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
print("--- df1 ---\n", df1)
print("\n--- df2 ---\n", df2)

# 将它们相加
result_df = df1 + df2
print("\n--- df1 + df2 ---\n", result_df)
```
**输出结果解读:**
结果的行索引是 `df1` 和 `df2` 行索引的并集，列索引也是它们列索引的并集。只有在**行和列都对得上**的位置才能正确计算，否则就是 `NaN`。例如，`Ohio` 行和 `b` 列在两者中都存在，所以 `0.0 + 3.0 = 3.0`。而 `c` 列只在 `df1` 中有，所以结果中 `c` 列全是 `NaN`。

##### **3. 在算术方法中填充值**

在对不同索引的对象进行运算时，有时我们不希望结果是 `NaN`，而是希望用一个特殊值（比如 0）来填充。这时可以使用算术方法（如 `add`）并传入 `fill_value` 参数。

```python
# In [165] - [171]
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df2.loc[1, 'b'] = np.nan # 人为制造一个 NaN

print("--- df1 ---\n", df1)
print("\n--- df2 (有 NaN) ---\n", df2)

# 普通加法，NaN 传播
print("\n--- df1 + df2 (普通加法) ---\n", df1 + df2)

# 使用 add 方法并指定 fill_value=0
# 这意味着在计算前，会先把 df1 和 df2 中所有 NaN 的地方，以及在对齐过程中出现的缺失位置，都当作 0 来处理。
filled_add = df1.add(df2, fill_value=0)
print("\n--- df1.add(df2, fill_value=0) ---\n", filled_add)
```

**算术方法列表 (表5-5)**
Pandas 为所有常见算术运算都提供了相应的方法。每个方法都有一个以 `r` 开头的“反向”版本（例如 `sub` 和 `rsub`）。`df1.sub(df2)` 相当于 `df1 - df2`，而 `df1.rsub(df2)` 相当于 `df2 - df1`。

| 方法                | 说明            |
| :------------------ | :-------------- |
| `add`, `radd`       | 用于加法 (+)    |
| `sub`, `rsub`       | 用于减法 (-)    |
| `div`, `rdiv`       | 用于除法 (/)    |
| `floordiv`, `rfloordiv` | 用于整除 (//) |
| `mul`, `rmul`       | 用于乘法 (*)    |
| `pow`, `rpow`       | 用于指数 (**)   |

##### **4. `DataFrame` 和 `Series` 之间的运算 (广播 Broadcasting)**

当 `DataFrame` 和 `Series` 进行运算时，Pandas 会进行一种叫做**广播 (broadcasting)** 的操作。默认情况下，它会将 `Series` 的索引匹配到 `DataFrame` 的**列**，然后沿着**行**向下广播。

```python
# In [179] - [183]
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
# 取出第一行，它是一个 Series
series = frame.iloc[0]

print("--- DataFrame ---\n", frame)
print("\n--- Series (第一行) ---\n", series)

# DataFrame - Series
# series 的索引 ('b', 'd', 'e') 会匹配 frame 的列。
# 然后，这个 series 会被"广播"到 frame 的每一行上进行减法运算。
# 相当于 frame 的每一行都减去 series。
result = frame - series
print("\n--- frame - series (广播) ---\n", result)
```

如果你希望匹配 `DataFrame` 的**行索引**并在**列**上进行广播，则必须使用算术方法并指定 `axis`。

```python
# In [186] & [189]
# 取出 'd' 列
series3 = frame['d']
print("--- 'd' 列 Series ---\n", series3)

# 希望 frame 的每一列都减去 series3
# 我们需要匹配行索引 (axis='index' or axis=0)
result_axis = frame.sub(series3, axis='index')
print("\n--- 按行匹配进行减法 ---\n", result_axis)
```

---

### **5.3 汇总和计算描述统计**

Pandas 对象拥有一组常用的数学和统计方法，它们大部分都属于**约简 (reduction)** 或**汇总统计**。

##### **1. 常用统计方法**

这些方法默认在**列**上进行计算 (axis=0)。

```python
# In [230] & [231]
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
print("--- 原始 DataFrame ---\n", df)

# In [232]
# .sum() 计算每列的总和，默认跳过 NaN
print("\n--- 每列的总和 .sum() ---\n", df.sum())

# In [233]
# 指定 axis=1 或 axis='columns'，则按行计算
print("\n--- 每行的总和 .sum(axis=1) ---\n", df.sum(axis=1))

# .mean() 计算平均值
print("\n--- 每列的平均值 .mean() ---\n", df.mean())

# .idxmax() 返回每列最大值所在的索引
print("\n--- 每列最大值的索引 .idxmax() ---\n", df.idxmax())

# .cumsum() 计算累计和
print("\n--- 每列的累计和 .cumsum() ---\n", df.cumsum())
```

**`skipna` 参数**
大多数这类方法都有一个 `skipna` 参数，默认为 `True`。如果设为 `False`，则在计算时**不排除 `NaN`**。只要有一个 `NaN`，计算结果就是 `NaN`。

```python
# In [234]
# 计算行平均值，但不跳过 NaN。'a' 行和 'c' 行因为有 NaN，所以结果也是 NaN
print("\n--- df.mean(axis=1, skipna=False) ---\n", df.mean(axis=1, skipna=False))
```

##### **2. `describe()` 方法**

`describe()` 方法是一个非常方便的函数，可以一次性产生多个汇总统计信息。

```python
# In [237]
# 对数值型数据
print("\n--- df.describe() ---\n", df.describe())

# In [238] & [239]
# 对非数值型数据 (如字符串)，describe 会给出另一种汇总信息
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
print("\n--- 非数值型 Series 的 describe() ---\n", obj.describe())
```

**描述和汇总统计方法列表 (表5-8)**

| 方法           | 说明                                       |
| :------------- | :----------------------------------------- |
| `count`        | 非 NA 值的数量                             |
| `describe`     | 针对 Series 或 DataFrame 的列计算汇总统计 |
| `min`, `max`   | 计算最小值和最大值                         |
| `argmin`, `argmax` | 计算能获取到最小值和最大值的索引位置（整数） |
| `idxmin`, `idxmax` | 计算能获取到最小值和最大值的索引值（标签） |
| `quantile`     | 计算样本的分位数（0 到 1）                 |
| `sum`          | 值的总和                                   |
| `mean`         | 值的平均数                                 |
| `median`       | 值的算术中位数（50%分位数）                |
| `mad`          | 根据平均值计算平均绝对离差                 |
| `var`          | 样本值的方差                               |
| `std`          | 样本值的标准差                             |
| `skew`         | 样本值的偏度（三阶矩）                     |
| `kurt`         | 样本值的峰度（四阶矩）                     |
| `cumsum`       | 样本值的累计和                             |
| `cummin`, `cummax` | 样本值的累计最小值和累计最大值             |
| `cumprod`      | 样本值的累计积                             |
| `diff`         | 计算一阶差分（对时间序列很有用）           |
| `pct_change`   | 计算百分数变化                             |

---

##### **3. 相关系数与协方差**

`corr()` 和 `cov()` 方法可以分别计算相关系数和协方差矩阵。

```python
# 由于 pandas-datareader 可能不稳定，我们自己创建一些模拟数据
returns_data = {
    'AAPL': np.random.randn(100) / 100,
    'GOOG': np.random.randn(100) / 100,
    'IBM': np.random.randn(100) / 100,
    'MSFT': np.random.randn(100) / 100,
}
returns = pd.DataFrame(returns_data)

# In [244] & [245]
# 计算两列之间的相关系数和协方差
msft_ibm_corr = returns['MSFT'].corr(returns['IBM'])
msft_ibm_cov = returns['MSFT'].cov(returns['IBM'])
print(f"MSFT 与 IBM 的相关系数: {msft_ibm_corr:.4f}")
print(f"MSFT 与 IBM 的协方差: {msft_ibm_cov:.6f}")

# In [247] & [248]
# 直接对 DataFrame 调用 .corr() 或 .cov() 会返回一个完整的矩阵
print("\n--- 相关系数矩阵 ---\n", returns.corr())
print("\n--- 协方差矩阵 ---\n", returns.cov())
```

##### **4. `corrwith` 方法**

`corrwith` 方法可以计算一个 `DataFrame` 的列或行与另一个 `Series` 或 `DataFrame` 之间的相关系数。

```python
# In [249]
# 计算每一列与 'IBM' 列的相关系数
corr_with_ibm = returns.corrwith(returns.IBM)
print("\n--- 各列与 IBM 的相关系数 ---\n", corr_with_ibm)
```

---

##### **5. 唯一值、值计数以及成员资格**

这些是一类用于从一维 `Series` 中抽取信息的方法。

```python
# In [251]
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])

# In [252]
# .unique() 返回 Series 中的唯一值数组，顺序是发现顺序
uniques = obj.unique()
print("唯一值:", uniques)

# In [254]
# .value_counts() 返回一个 Series，索引是唯一值，值是每个值出现的频率，并按频率降序排列
counts = obj.value_counts()
print("\n值计数:\n", counts)

# In [257]
# .isin() 用于判断矢量化集合的成员资格，返回一个布尔 Series
mask = obj.isin(['b', 'c'])
print("\n--- isin(['b', 'c']) 的掩码 ---\n", mask)
print("\n--- 使用掩码过滤 ---\n", obj[mask])
```

**唯一值、值计数、成员资格方法 (表5-9)**

| 方法           | 说明                                                                     |
| :------------- | :----------------------------------------------------------------------- |
| `isin`         | 计算一个表示 “Series 各值是否包含于传入的值序列中” 的布尔型数组。       |
| `match`        | 计算一个数组中各值到另一个不同值数组的整数索引。（已不常用，推荐 `get_indexer`） |
| `unique`       | 计算 Series 中的唯一值数组，按发现的顺序返回。                           |
| `value_counts` | 返回一个 Series，其索引为唯一值，其值为频率，按计数值降序排列。          |
