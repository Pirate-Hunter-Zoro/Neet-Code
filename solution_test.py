from typing import List
import unittest

from solution import Solution

class SolutionTest(unittest.TestCase):
    
    
    def results_helper(self, f, inputs: List[any], expected_outputs: List[any]):
        for input, output in zip(inputs, expected_outputs):
            self.assertTrue(f(input) == output)


    def test_topologicalSort(self):
        sol = Solution()

        class Input:
            def __init__(self, n: int, edges: List[List[int]]):
                self.n = n
                self.edges = edges
        
        inputs = [Input(3, [[0,1], [1,2], [2,0]])]
        expected_outputs = [[]]
        
        f = lambda x : sol.topologicalSort(x.n, x.edges)
        
        self.results_helper(f, inputs, expected_outputs)
            
            
    def test_wordBreak(self):
        sol = Solution()
        
        class Input:
            def __init__(self, s: str, wordDict: List[str]):
                self.s = s
                self.wordDict = wordDict
        
        inputs = [Input("neetcode", ["neet","code"]), Input("applepenapple", ["apple","pen","ape"]), Input("catsincars", ["cats","cat","sin","in","car"])]
        expected_outputs = [True, True, False]
        
        f = lambda x : sol.wordBreak(x.s, x.wordDict)
            
        self.results_helper(f, inputs, expected_outputs)
            
            
    def test_subsets(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int]):
                self.nums = nums
                
        inputs = [Input([1,2,3])]
        expected_outputs = [[[],[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]]
        
        f = lambda x : sol.subsets(x.nums)
        
        self.results_helper(f, inputs, expected_outputs)
            
            
    def test_canPartition(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int]):
                self.nums = nums
        
        inputs = [Input([1,2,3,4]), Input([1,2,3,4,5]), Input([1,1]), Input([1,2,5])]
        expected_outputs = [True, False, True, False]
        
        f = lambda x : sol.canPartition(x.nums)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_lengthOfLIS(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int]):
                self.nums = nums
                
        inputs = [Input([9,1,4,2,3,3,7]), Input([0,3,1,3,2,3])]
        expected_outputs = [4, 4]
        
        f = lambda x : sol.lengthOfLIS(x.nums)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_numDistict(self):
        sol = Solution()
        
        class Input:
            def __init__(self, s: str, t: str):
                self.s = s
                self.t = t
                
        inputs = [Input("caaat", "cat"), Input("xxyxy", "xy")]
        expected_outputs = [3, 5]
        
        f = lambda x : sol.numDistinct(x.s, x.t)
        
        self.results_helper(f, inputs, expected_outputs)
        
    
    def test_minInterval(self):
        sol = Solution()
        
        class Input:
            def __init__(self, intervals: List[List[int]], queries: List[int]):
                self.intervals = intervals
                self.queries = queries
                
        inputs = [Input(intervals = [[1,3],[2,3],[3,7],[6,6]], queries = [2,3,1,7,6,8]), Input(intervals=[[4,5],[5,8],[1,9],[8,10],[1,6]], queries=[7,9,3,9,3])]
        expected_outputs = [[2,2,3,5,1,-1], [4,3,6,3,6]]
        
        f = lambda x: sol.minInterval(x.intervals, x.queries)
        
        self.results_helper(f, inputs, expected_outputs)