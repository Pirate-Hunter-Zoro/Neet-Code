from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None, values: List[int]=None):
        # Initialize the attributes
        self.left = None
        self.right = None
        self.val = None
        
        if values != None:
            # One way to initialize a tree node
            self.val = values[0]
            if len(values) < 2:
                return
            else:
                if len(values) == 2:
                    self.left = TreeNode(val=values[1])
                else:
                    current_level = [self]
                    current_idx = 0
                    while len(current_level) > 0:
                        # We are still ready to make more layers
                        current_level_idx = 0
                        next_level_value_indices = range(current_idx + len(current_level), current_idx + len(current_level) + 2*len(current_level))
                        next_level = []
                        # Now to fill in the next depth of nodes
                        
        else:
            # Other way to initialize a tree node
            self.val = val
            self.left = left
            self.right = right