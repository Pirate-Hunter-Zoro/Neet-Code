from typing import Optional

from implementation.linked_list import Stack
from implementation.tree_node import TreeNode


class Codec:
    
    # Encodes a tree to a single string.
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialization is the process of converting an in-memory structure into a sequence of bits so that it can be stored or sent across a network to be reconstructed later in another computer environment.

        Args:
            root (Optional[TreeNode]): tree to serialize

        Returns:
            str: serialization of said tree
        """
        if root == None:
            return ""
        else:
            result = str(root.val)
            if root.left != None:
                result += "L(" + self.serialize(root.left) + ")"
            if root.right != None:
                result += "R(" + self.serialize(root.right) + ")"
            return result
        
    # Decodes your encoded data to tree.
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure. 
        There is no additional restriction on how your serialization/deserialization algorithm should work.

        Args:
            data (str): serialized binary tree

        Returns:
            Optional[TreeNode]: tree yielded from the serialization
        """
        if data == "":
            return None
        else:
            first_child_idx = data.find("L")
            first_parentheses_idx = data.find("(")
            if first_parentheses_idx == -1:
                # No child 
                return TreeNode(val=int(data))
            elif first_child_idx == -1 or first_child_idx > first_parentheses_idx:
                # No left child - only right child is present
                # Format is R(...), where the ... is the right subtree
                first_child_idx = data.find("R")
                root = TreeNode(val=int(data[:first_child_idx]))
                root.right = self.deserialize(data[first_child_idx+2:len(data)-1])
                return root
            else:
                # Left child present and MAYBE right child present
                root = TreeNode(val=int(data[:first_child_idx]))
                end_of_first_child_data = first_parentheses_idx
                index_stack = Stack()
                index_stack.push(end_of_first_child_data)
                end_of_first_child_data += 1
                while len(index_stack) > 0:
                    next_char = data[end_of_first_child_data]
                    if next_char == ")":
                        index_stack.pop()
                    elif next_char == "(":
                        index_stack.push(end_of_first_child_data)
                    end_of_first_child_data += 1
                # Left child data bounds obtained
                root.left = self.deserialize(data=data[first_parentheses_idx+1:end_of_first_child_data-1])
                if end_of_first_child_data < len(data):
                    # There is a right child too
                    root.right = self.deserialize(data=data[end_of_first_child_data+2:len(data)-1])
                return root