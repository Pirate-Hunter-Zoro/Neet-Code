import unittest

from implementation.median_finder import MedianFinder

class MedianFinderTest(unittest.TestCase):
    
    def test_MedianFinder(self):
        
        median_finder = MedianFinder()
        
        median_finder.addNum(1)
        self.assertEqual(median_finder.findMedian(), 1.0)
        
        median_finder.addNum(3)
        self.assertEqual(median_finder.findMedian(), 2.0)
        
        median_finder.addNum(2)
        self.assertEqual(median_finder.findMedian(), 2.0)
        
        
        median_finder = MedianFinder()
        
        median_finder.addNum(5)
        median_finder.addNum(3)
        self.assertEqual(median_finder.findMedian(), 4.0)
        
        median_finder.addNum(7)
        self.assertEqual(median_finder.findMedian(), 5.0)
        
        median_finder.addNum(2)
        self.assertEqual(median_finder.findMedian(), 4.0)
