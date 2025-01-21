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
                            if current_idx >= len(values):
                                break
                            
                            # Right child
                            right_val = values[current_idx]
                            if right_val != None:
                                node.right = TreeNode(val=right_val)
                            next_level.append(node.right)
                            current_idx += 1
                            if current_idx >= len(values):
                                break
                                
                        # Onto the next layer
                        current_level = next_level
        else:
            # Other way to initialize a tree node
            self.val = val
            self.left = left
            self.right = right
            
    def __eq__(self, other) -> bool:
        """Overloader for determining if this tree node is the same as the other

        Args:
            other: other tree node
        """
        if other == None:
            return False
        elif type(other) != type(self):
            return False
        else:
            return self.__equals(other=other)
    
    def __equals(self, other) -> bool:
        if self.val != other.val:
            return False
        elif self.left == None and other.left != None:
            return False
        elif self.right == None and other.right != None:
            return False
        elif other.left == None and self.left != None:
            return False
        elif other.right == None and self.right != None:
            return False
        else:
            if self.left != None and not self.left.__equals(other.left):
                return False
            if self.right != None and not self.right.__equals(other.right):
                return False
            return True