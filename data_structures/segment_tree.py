from typing import List


class SegmentTreeNode:
    def __init__(self, nums: list[int], left_idx: int, right_idx: int):
        self.__left_child = None
        self.__right_child = None
        mid = (left_idx + right_idx) // 2
        self.__idx = mid
        self.__left_idx = left_idx
        self.__right_idx = right_idx
        self.__value = nums[mid]
        self.__sum = nums[mid]
        if left_idx < mid:
            self.__left_child = SegmentTreeNode(nums=nums, left_idx=left_idx, right_idx=mid-1)
            self.__sum += self.__left_child.__sum
        if right_idx > mid:
            self.__right_child = SegmentTreeNode(nums=nums, left_idx=mid+1, right_idx=right_idx)
            self.__sum += self.__right_child.__sum
    
    def change_element(self, idx: int, val: int):
        """Helper method to update the underlying value at a given index

        Args:
            idx (int): index where we need to change the value
            val (int): value to change the given element
        """
        if idx == self.__idx:
            self.__sum -= self.__value
            self.__value = val
            self.__sum += val
        elif idx < self.__idx:
            # Go to left child
            self.__sum -= self.__left_child.__value
            self.__left_child.change_element(idx=idx, val=val)
        else:
            # Go to right child
            self.__sum -= self.__right_child.__value
            self.__right_child.change_element(idx=idx, val=val)
        self.__sum += val 
        
    def get_sum(self, left_idx: int, right_idx: int) -> int:
        """Return the sum in the range of the left and right indices.
        Note that this function assumes [left_idx, right_idx] is contained in [self.__left_idx, self.__right_idx]

        Args:
            left_idx (int): left bound
            right_idx (int): right bound

        Returns:
            int: sum in that range (inclusive)
        """
        sum = 0
        if self.__idx >= left_idx and self.__idx <= right_idx:
            # This node will be included in sum
            sum += self.__value
        if left_idx < self.__idx:
            # Part of the left child subtree will be included in this sum
            sum += self.__left_child.get_sum(left_idx=left_idx, right_idx=min(self.__idx-1, right_idx))
        if right_idx > self.__idx:
            # Part of the right child subtree will be included in this sum
            sum += self.__right_child.get_sum(left_idx=max(left_idx, self.__idx+1), right_idx=right_idx)
        return sum

class SegmentTree:
    
    def __init__(self, nums: List[int]):
        """SegmentTree(int[] arr) will initialize a segment tree based on the given array arr. 
        You can assume that the array arr is non-empty.

        Args:
            nums (List[int]): numbers to go in the segment tree
        """
        self.__root = SegmentTreeNode(nums=nums, left_idx=0, right_idx=len(nums)-1)
    
    def update(self, index: int, val: int) -> None:
        """void update(int idx, int val) will update the element at index idx in the original array to be val. 
        You can assume that 0 <= idx < arr.length.

        Args:
            index (int): index to update the value
            val (int): value to give said index
        """
        self.__root.change_element(idx=index, val=val)
    
    def query(self, L: int, R: int) -> int:
        """int query(int l, int r) will return the sum of all elements in the range [l, r] inclusive. 
        You can assume that 0 <= l <= r < arr.length.

        Args:
            L (int): left bound
            R (int): right bound

        Returns:
            int: sum of elements in left bound to right bound
        """
        return self.__root.get_sum(left_idx=L, right_idx=R)