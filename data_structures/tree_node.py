from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None, values: List[int]=None):
        if values != None:
            # One way to initialize a tree node
            pass
        else:
            # Other way to initialize a tree node
            self.val = val
            self.left = left
            self.right = right