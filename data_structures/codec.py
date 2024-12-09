from typing import Optional

from data_structures.tree_node import TreeNode


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
            first_paren_idx = data.find("(")
            if first_paren_idx == -1:
                # No child 
                return TreeNode(val=int(data))
            elif first_child_idx == -1 or first_child_idx > first_paren_idx:
                # No left child - only right child is present
                # Format is R(...), where the ... is the right subtree
                first_child_idx = data.find("R")
                root = TreeNode(val=int(data[:first_child_idx]))
                root.right = self.deserialize(data[first_child_idx+2:len(data)-1])
            else:
                # Left child present and MAYBE right child present
                root = TreeNode(val=int(data[:first_child_idx]))
                end_of_first_child_data = first_paren_idx
                
                right_child_idx = data.find("R")
                if right_child_idx == -1 or right_child_idx > first_paren_idx:
                    # Only left child
                    # Format is L(...), where the ... is the left subtree
                    root.left = self.deserialize(data=data[first_child_idx+2:len(data)-1])
                else:
                    # Both children present
                    # Format is L(...)R(...), where each ... is a subtree
                    root.left = self.deserialize(data=data[first_child_idx+2:right_child_idx-1])
                    root.right = self.deserialize(data=data[right_child_idx+2:len(data)-1])
                return root