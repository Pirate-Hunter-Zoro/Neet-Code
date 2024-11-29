from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None, values: List[int]=None):
        self.val = val
        self.left = left
        self.right = right
        
        if values != None:
            # Other way to initialize a tree node
            pass