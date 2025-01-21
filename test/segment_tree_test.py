import unittest

import unittest

from implementation.segment_tree import SegmentTree

class SegmentTreeTest(unittest.TestCase):
    
    
    def test_segment_tree(self):
        seg = SegmentTree(nums=[1, 2, 3, 4, 5])

        self.assertTrue(seg.query(L=0, R=2) == 6)
        self.assertTrue(seg.query(L=2, R=4) == 12)
        seg.update(index=3, val=0)
        self.assertTrue(seg.query(L=2, R=4) == 8)