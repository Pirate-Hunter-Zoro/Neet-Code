import unittest

from implementation.disjoint_set import UnionFind

class UnionFindTest(unittest.TestCase):
    
    
    def test_disjoint_set_union(self):
        disjoint_set = UnionFind(n=10)
        
        self.assertFalse(disjoint_set.isSameComponent(x=1, y=3))
        self.assertTrue(disjoint_set.union(x=1, y=2))
        self.assertTrue(disjoint_set.union(x=2, y=3))
        self.assertEqual(disjoint_set.getNumComponents(), 8)
        self.assertTrue(disjoint_set.isSameComponent(x=1, y=3))