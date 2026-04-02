# 允许在类定义内部使用尚未完全定义的类型作为提示，这对于节点类至关重要
from __future__ import annotations

class DoublyListNode:
    """双向链表中的一个节点。"""
    def __init__(self, val: int):
        """
        初始化一个新节点。

        Args:
            val (int): 存储在节点中的整数值。
        """
        self.val: int = val
        # prev 指向前一个节点，可以是节点对象或 None
        self.prev: DoublyListNode | None = None
        # next 指向后一个节点，可以是节点对象或 None
        self.next: DoublyListNode | None = None

class DoublyLinkedList:
    """
    一个完整的双向链表实现，使用头尾哨兵节点简化操作。
    """
    def __init__(self):
        """
        初始化一个空的双向链表。
        
        为了简化插入和删除的边界条件，我们创建了两个哨兵节点：
        - self._head: 位于链表最前端的虚拟头节点。
        - self._tail: 位于链表最末端的虚拟尾节点。
        
        初始状态下，链表为空，结构为: head <-> tail
        """
        self._head = DoublyListNode(0)  # 哨兵头节点
        self._tail = DoublyListNode(0)  # 哨兵尾节点
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def get_size(self) -> int:
        """返回链表中的元素数量。"""
        return self._size

    def is_empty(self) -> bool:
        """检查链表是否为空。"""
        return self._size == 0

    def addLast(self, val: int) -> None:
        """
        在链表尾部添加一个新节点（在 self._tail 哨兵之前）。
        """
        new_node = DoublyListNode(val)
        last_node = self._tail.prev
        
        # 步骤 1: last_node 和 new_node 建立连接
        last_node.next = new_node
        new_node.prev = last_node
        
        # 步骤 2: new_node 和 tail 建立连接
        new_node.next = self._tail
        self._tail.prev = new_node
        
        self._size += 1

    def addFirst(self, val: int) -> None:
        """
        在链表头部添加一个新节点（在 self._head 哨兵之后）。
        
        """
        # 利用 addLast 的逻辑，在其前一个节点后插入即可
        # 这里为了清晰，我们重新实现
        new_node = DoublyListNode(val)
        first_node = self._head.next

        # 步骤 1: head 和 new_node 建立连接
        self._head.next = new_node
        new_node.prev = self._head

        # 步骤 2: new_node 和 first_node 建立连接
        new_node.next = first_node
        first_node.prev = new_node
        
        self._size += 1

    def removeLast(self) -> int:
        """
        删除并返回链表尾部的节点值（self._tail 哨兵之前的节点）。
        
        """
        if self.is_empty():
            raise IndexError("Cannot remove from an empty list.")
            
        node_to_remove = self._tail.prev
        prev_node = node_to_remove.prev
        
        # 将 prev_node 和 tail 直接连接，跳过被删除的节点
        prev_node.next = self._tail
        self._tail.prev = prev_node
        
        self._size -= 1
        return node_to_remove.val

    def removeFirst(self) -> int:
        """
        删除并返回链表头部的节点值（self._head 哨兵之后的节点）。
        """
        if self.is_empty():
            raise IndexError("Cannot remove from an empty list.")
            
        node_to_remove = self._head.next
        next_node = node_to_remove.next
        
        # 将 head 和 next_node 直接连接
        self._head.next = next_node
        next_node.prev = self._head
        
        self._size -= 1
        return node_to_remove.val

    def __str__(self) -> str:
        """
        提供链表的字符串表示，方便打印和调试。
        """
        if self.is_empty():
            return "DoublyLinkedList: []"
        
        vals = []
        current = self._head.next
        while current != self._tail:
            vals.append(str(current.val))
            current = current.next
        return "DoublyLinkedList: [" + " <-> ".join(vals) + "]"


# --- 使用示例 ---
if __name__ == "__main__":
    dll = DoublyLinkedList()
    print(f"初始化: {dll}, 大小: {dll.get_size()}")

    print("\n--- 在尾部添加元素 ---")
    dll.addLast(10)
    dll.addLast(20)
    dll.addLast(30)
    print(f"添加后: {dll}, 大小: {dll.get_size()}")

    print("\n--- 在头部添加元素 ---")
    dll.addFirst(5)
    print(f"添加后: {dll}, 大小: {dll.get_size()}")

    print("\n--- 删除尾部元素 ---")
    removed_val = dll.removeLast()
    print(f"删除的值: {removed_val}")
    print(f"删除后: {dll}, 大小: {dll.get_size()}")

    print("\n--- 删除头部元素 ---")
    removed_val = dll.removeFirst()
    print(f"删除的值: {removed_val}")
    print(f"删除后: {dll}, 大小: {dll.get_size()}")

    print("\n--- 连续删除 ---")
    dll.removeLast()
    dll.removeFirst()
    print(f"全部删除后: {dll}, 大小: {dll.get_size()}")

    # 尝试在空链表上删除会触发异常
    try:
        dll.removeFirst()
    except IndexError as e:
        print(f"\n捕获到预期异常: {e}")