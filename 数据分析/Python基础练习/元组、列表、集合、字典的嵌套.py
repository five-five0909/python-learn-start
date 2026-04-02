# ============================
# 一、列表推导式（List Comprehensions）
# 用于快速生成列表，语法简洁高效
# ============================

# 示例1：生成1到10的平方数列表
squares = [x ** 2 for x in range(1, 11)]  # range(1,11) → 1到10
print("1到10的平方数列表:", squares)
# 输出: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# 示例2：筛选1到20中的偶数，并计算它们的平方
even_squares = [x ** 2 for x in range(1, 21) if x % 2 == 0]  # 只选择偶数
print("1到20中偶数的平方:", even_squares)
# 输出: [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]

# 示例3：两个列表的笛卡尔积（所有可能的组合）
colors = ['红', '绿', '蓝']
sizes = ['S', 'M', 'L']
combinations = [(color, size) for color in colors for size in sizes]  # 嵌套循环
print("颜色和尺码的所有组合:", combinations)
# 输出: [('红', 'S'), ('红', 'M'), ('红', 'L'), ('绿', 'S'), ..., ('蓝', 'L')]


# ============================
# 二、集合推导式（Set Comprehensions）
# 用于生成不重复元素的集合
# ============================

# 示例1：生成1到10中偶数的平方，并去重（虽然这里不会重复，但展示了用法）
unique_even_squares = {x ** 2 for x in range(1, 11) if x % 2 == 0}
print("1到10中偶数的平方（去重集合）:", unique_even_squares)
# 输出类似: {4, 16, 36, 64, 100}（集合无序）

# 示例2：从一段文字中提取所有不重复的小写字母
sentence = "Hello World!"
unique_letters = {char.lower() for char in sentence if char.isalpha()}  # 只保留字母并转小写
print("句子中的不重复小写字母:", unique_letters)
# 输出类似: {'h', 'e', 'l', 'o', 'w', 'r', 'd'}


# ============================
# 三、字典推导式（Dict Comprehensions）
# 用于动态生成字典，灵活处理键值对
# ============================

# 示例1：将两个列表组合成键值对字典
keys = ['a', 'b', 'c']
values = [1, 2, 3]
my_dict = {keys[i]: values[i] for i in range(len(keys))}  # 按索引配对
print("列表组合生成的键值对字典:", my_dict)
# 输出: {'a': 1, 'b': 2, 'c': 3}

# 示例2：从一个字典中筛选出值大于10的键值对，生成新字典
original_dict = {'apple': 5, 'banana': 12, 'cherry': 8}
filtered_dict = {k: v for k, v in original_dict.items() if v > 10}  # 只保留值>10的项
print("值大于10的筛选后字典:", filtered_dict)
# 输出: {'banana': 12}

# 示例3：将字符串列表转换为字典，键为单词，值为(长度, 大写形式)
words = ["Python", "is", "awesome"]
word_info = {word: (len(word), word.upper()) for word in words}  # 键是单词，值是元组(长度, 大写)
print("单词信息字典:", word_info)
# 输出: {'Python': (6, 'PYTHON'), 'is': (2, 'IS'), 'awesome': (7, 'AWESOME')}
