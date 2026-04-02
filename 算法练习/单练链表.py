# 允许在类定义内部使用尚未完全定义的类型作为提示，这对于节点类至关重要
from __future__ import annotations


class ListNode:
    """双向链表中的一个节点。"""

    def __init__(self, val: int):
        """
        初始化一个新节点。

        Args:
            val (int): 存储在节点中的整数值。
        """
        self.val: int = val
        # next 指向后一个节点，可以是节点对象或 None
        self.next: ListNode | None = None


class LinkedList:
    def __init__(self):
        """
        初始化一个空的单链链表
        """
        self._head = ListNode(0)  # 头节点
        self._tail = self._head  # 尾节点
        self._size = 0

    def get_size(self) -> int:
        """返回链表中的元素数量。"""
        return self._size

    def is_empty(self) -> bool:
        """检查链表是否为空。"""
        return self._size == 0

    def addLast(self, val: int) -> None:
        """
        在链表尾部添加一个新节点
        """
        new_node = ListNode(val)
        self._tail.next = new_node
        self._tail = new_node
        self._size += 1

    def addFirst(self, val: int) -> None:
        """
        在链表头部添加一个新节点
        """
        new_node = ListNode(val)
        # 判断链表是否为空
        if self._tail is None:
            self._tail = new_node

        new_node.next = self._head.next
        self._head.next = new_node

        self._size += 1

    def removeLast(self) -> int:
        """
        删除并返回链表尾部的节点值
        """
        if self.is_empty():
            raise IndexError("Cannot remove from an empty list.")

        # 找尾节点的前驱节点
        cur_node = self._head
        while cur_node is not self._tail:
            cur_node = cur_node.next
        node_to_remove = self._tail
        prev_node = cur_node
        prev_node.next = None
        self._tail = prev_node

        self._size -= 1
        return node_to_remove.val

    def removeFirst(self) -> int:
        """
        删除并返回链表头部的节点值
        """
        if self.is_empty():
            raise IndexError("Cannot remove from an empty list.")

        node_to_remove = self._head.next
        self._head.next = self._head.next.next
        self._size -= 1
        if self.is_empty():
            self._tail = self._head
        return node_to_remove.val

    def __str__(self) -> str:
        """提供链表的字符串表示，方便打印和调试。"""
        if self.is_empty():
            return "LinkedList: []"

        vals = []
        current = self._head.next
        while current:
            vals.append(str(current.val))
            current = current.next
        return "LinkedList: [" + " -> ".join(vals) + "]"
