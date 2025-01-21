import unittest

from implementation.codec import Codec
from implementation.tree_node import TreeNode


class CodecTest(unittest.TestCase):
    
    def test_codec(self):
        codec = Codec()
        node_lists = [[1,None,3,4,5,6,7], [1,2,3,None,None,4,5], [1,2,3,None,None,None,4], [1], [2,1,3], [4,3,5,2,None], [2,1,1,3,None,1,5], [1,2,-1,3,4], [4,7], [4,None,7], [1,2,3,4,5,6,7], [1,2,3,None,None,4]]
        trees = [TreeNode(values=node_list) for node_list in node_lists]
        for tree in trees:
            self.assertTrue(codec.deserialize(codec.serialize(tree)) == tree)
