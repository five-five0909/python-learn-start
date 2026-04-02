class CycleArray:
    def __init__(self, size):
        self.size = size
        # size这么多个空格创建出来
        self.arr = [None] * size
        self.count = 0
        # 此时start==end：就表示数组为空
        self.start = 0
        # 即 end 指向最后一个有效元素的下一个位置索引
        # 理论上，你可以随意设计区间的开闭，但一般设计为左闭右开区间是最方便处理的。
        self.end = 0

    def resize(self, newSize):
        new_arr = [None] * newSize
        new_arr.extend(self.arr)
        self.arr = new_arr
        self.size = newSize
        # 重置 start 和 end 指针
        self.start = 0
        self.end = self.count
        self.size = newSize
        
    # 在数组头部添加元素，时间复杂度 O(1)
    def add_first(self, val):
        # 当数组满时，扩容为原来的两倍
        if self.is_full():
            self.resize(self.size * 2)
        # 因为 start 是闭区间，所以先左移，再赋值
        self.start = (self.start - 1 + self.size) % self.size
        self.arr[self.start] = val
        self.count += 1

    # 删除数组头部元素，时间复杂度 O(1)
    def remove_first(self):
        if self.is_empty():
            raise Exception("Array is empty")
        # 因为 start 是闭区间，所以先赋值，再右移
        self.arr[self.start] = None
        self.start = (self.start + 1) % self.size
        self.count -= 1
        # 如果数组元素数量减少到原大小的四分之一，则减小数组大小为一半
        if self.count > 0 and self.count == self.size // 4:
            self.resize(self.size // 2)

    # 在数组尾部添加元素，时间复杂度 O(1)
    def add_last(self, val):
        if self.is_full():
            self.resize(self.size * 2)
        # 因为 end 是开区间，所以是先赋值，再右移
        self.arr[self.end] = val
        self.end = (self.end + 1) % self.size
        self.count += 1

    def remove_last(self, val):
        if self.is_empty():
            raise Exception("Array is empty")
        # 因为 end 是开区间，所以是先右移，再删除
        self.end = (self.end - 1) % self.size
        self.arr[self.end] = None
        self.count -= 1
        # 如果数组元素数量减少到原大小的四分之一，则减小数组大小为一半
        if self.count > 0 and self.count == self.size // 4:
            self.resize(self.size // 2)

    # 获取数组头部元素，时间复杂度 O(1)
    def get_first(self):
        if self.is_empty():
            raise Exception("Array is empty")
        return self.arr[self.start]

    # 获取数组尾部元素，时间复杂度 O(1)
    def get_last(self):
        if self.is_empty():
            raise Exception("Array is empty")
        # end 是开区间，指向的是下一个元素的位置，所以要减 1
        return self.arr[(self.end - 1 + self.size) % self.size]

    def is_full(self):
        return self.count == self.size

    def size(self):
        return self.count

    def is_empty(self):
        return self.count == 0
