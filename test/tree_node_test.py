import unittest

from implementation.tree_node import TreeNode


class TreeNodeTest(unittest.TestCase):
    
    def test_tree_node_construction(self):
        node_lists = [[1,2,3,None,None,4,5], [1,2,3,None,None,None,4], [1], [2,1,3], [4,3,5,2,None], [2,1,1,3,None,1,5], [1,2,-1,3,4], [4,7], [4,None,7], [1,2,3,4,5,6,7], [1,2,3,None,None,4]]
        
        expected_trees = [TreeNode(1, 
                                   TreeNode(2), 
                                   TreeNode(3, 
                                            TreeNode(4), 
                                            TreeNode(5))),
                          TreeNode(1,
                                   TreeNode(2),
                                   TreeNode(3, 
                                            right=TreeNode(4))),
                          TreeNode(1),
                          TreeNode(2,
                                   TreeNode(1),
                                   TreeNode(3)),
                          TreeNode(4,
                                   TreeNode(3,
                                            TreeNode(2)),
                                   TreeNode(5)),
                          TreeNode(2,
                                   TreeNode(1,
                                            TreeNode(3)),
                                   TreeNode(1,
                                            TreeNode(1),
                                            TreeNode(5))),
                          TreeNode(1,
                                   TreeNode(2,
                                            TreeNode(3),
                                            TreeNode(4)),
                                   TreeNode(-1)),
                          TreeNode(4,
                                   TreeNode(7)),
                          TreeNode(4,
                                   right=TreeNode(7)),
                          TreeNode(1,
                                   TreeNode(2,
                                            TreeNode(4),
                                            TreeNode(5),
                                            ),
                                   TreeNode(3,
                                            TreeNode(6),
                                            TreeNode(7),
                                            ),
                                   ),
                          TreeNode(1,
                                   TreeNode(2),
                                   TreeNode(3,
                                            TreeNode(4)))
                          ]
        
        for node_list, expected_tree in zip(node_lists, expected_trees):
            resulting_tree = TreeNode(values=node_list)
            self.assertTrue(resulting_tree == expected_tree)