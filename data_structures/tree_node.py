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
                    current_idx = 1
                    while current_idx < len(values):
                        # We are still ready to make more layers
                        next_level = []
                        # Now to fill in the next depth of nodes
                        for node in current_level:
                            if node == None: 
                                continue
                            
                            # Left child
                            left_val = values[current_idx]
                            if left_val != None:
                                node.left = TreeNode(val=left_val)
                            next_level.append(node.left)
                            current_idx += 1
                            
                            # Right child
                            if current_idx >= len(values):
                                break
                            else:
                                right_val = values[current_idx]
                                if right_val != None:
                                    node.right = TreeNode(val=right_val)
                                next_level.append(node.right)
                                current_idx += 1
                                
                        # Onto the next layer
                        current_level = next_level
        else:
            # Other way to initialize a tree node
            self.val = val
            self.left = left
            self.right = right