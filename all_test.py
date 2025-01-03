from typing import List, Optional
import unittest

from data_structures.codec import Codec
from data_structures.disjoint_set import UnionFind
from data_structures.tree_node import TreeNode
from solution import Solution

from typing import List
import unittest

from data_structures.segment_tree import SegmentTree

class SegmentTreeTest(unittest.TestCase):
    
    
    def test_segment_tree(self):
        seg = SegmentTree(nums=[1, 2, 3, 4, 5])

        self.assertTrue(seg.query(L=0, R=2) == 6)
        self.assertTrue(seg.query(L=2, R=4) == 12)
        seg.update(index=3, val=0)
        self.assertTrue(seg.query(L=2, R=4) == 8)


class UnionFindTest(unittest.TestCase):
    
    
    def test_disjoint_set_union(self):
        disjoint_set = UnionFind(n=10)
        
        self.assertFalse(disjoint_set.isSameComponent(x=1, y=3))
        self.assertTrue(disjoint_set.union(x=1, y=2))
        self.assertTrue(disjoint_set.union(x=2, y=3))
        self.assertEqual(disjoint_set.getNumComponents(), 8)
        self.assertTrue(disjoint_set.isSameComponent(x=1, y=3))


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


class CodecTest(unittest.TestCase):
    
    def test_codec(self):
        codec = Codec()
        node_lists = [[1,None,3,4,5,6,7], [1,2,3,None,None,4,5], [1,2,3,None,None,None,4], [1], [2,1,3], [4,3,5,2,None], [2,1,1,3,None,1,5], [1,2,-1,3,4], [4,7], [4,None,7], [1,2,3,4,5,6,7], [1,2,3,None,None,4]]
        trees = [TreeNode(values=node_list) for node_list in node_lists]
        for tree in trees:
            self.assertTrue(codec.deserialize(codec.serialize(tree)) == tree)


class SolutionTest(unittest.TestCase):
    
    
    def results_helper(self, f, inputs: List[any], expected_outputs: List[any]):
        for input, output in zip(inputs, expected_outputs):
            result = f(input)
            self.assertTrue(result == output)


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
        
    
    def test_minimumSpanningTreeKruskal(self):
        sol = Solution()
        
        class Input:
            def __init__(self, n:int, edges:List[List[int]]):
                self.n = n
                self.edges = edges
                
        inputs = [Input(n=5, edges=[[0,1,10], [0,2,3], [1,3,2], [2,1,4], [2,3,8], [2,4,2], [3,4,5]])]
        expected_outputs = [11]
        
        f = lambda x: sol.minimumSpanningTreeKruskal(n=x.n, edges=x.edges)
        
        self.results_helper(f, inputs, expected_outputs)
        
    def test_minimumSpanningTreePrim(self):
        sol = Solution()
        
        class Input:
            def __init__(self, n:int, edges:List[List[int]]):
                self.n = n
                self.edges = edges
                
        inputs = [Input(n=5, edges=[[0,1,10], [0,2,3], [1,3,2], [2,1,4], [2,3,8], [2,4,2], [3,4,5]])]
        expected_outputs = [11]
        
        f = lambda x: sol.minimumSpanningTreePrim(n=x.n, edges=x.edges)
        
        self.results_helper(f, inputs, expected_outputs)
        
    def test_shortestPath(self):
        sol = Solution()
        
        class Input:
            def __init__(self, n:int, edges:List[List[int]], src: int):
                self.n = n
                self.edges = edges
                self.src = src
        
        inputs = [Input(n = 5, edges = [[0,1,10], [0,2,3], [1,3,2], [2,1,4], [2,3,8], [2,4,2], [3,4,5]], src = 0)]
        
        expected_outputs = [{0:0, 1:7, 2:3, 3:9, 4:5}]
        
        f = lambda x : sol.shortestPath(n=x.n, edges=x.edges, src=x.src)
        
        self.results_helper(f, inputs, expected_outputs)
        
    def test_maximumPathSum(self):
        sol = Solution()
        
        class Input:
            def __init__(self, root: Optional[TreeNode]):
                self.root = root
                
        inputs = [Input(root=TreeNode(values=[1,2,3])), Input(root=TreeNode(values=[-15,10,20,None,None,15,5,-5]))]
        
        expected_outputs = [6, 40]
        
        f = lambda x : sol.maxPathSum(x.root)
        
        self.results_helper(f, inputs, expected_outputs)
        
    def test_maxCoins(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int]):
                self.nums = nums
                
        inputs = [Input([4,2,3,7])]
        
        expected_outputs = [143]
        
        f = lambda x : sol.maxCoins(nums=x.nums)
        
        self.results_helper(f, inputs, expected_outputs)
       
        
    def test_ladderLength(self):
        sol = Solution()
        
        class Input:
            def __init__(self, beginWord: str, endWord: str, wordList: List[str]):
                self.beginWord = beginWord
                self.endWord = endWord
                self.wordList = wordList
                
        inputs = [Input(beginWord = "cat", endWord = "sag", wordList = ["bat","bag","sag","dag","dot"]), Input(beginWord = "cat", endWord = "sag", wordList = ["bat","bag","sat","dag","dot"])]
        
        expected_outputs = [4, 0]
        
        f = lambda x : sol.ladderLength(beginWord=x.beginWord, endWord=x.endWord, wordList=x.wordList)
        
        self.results_helper(f, inputs, expected_outputs)
        
    
    def test_findItinerary(self):
        sol = Solution()
        
        class Input:
            def __init__(self, tickets: List[List[str]]):
                self.tickets = tickets
                
        inputs = [Input(tickets = [["BUF","HOU"],["HOU","SEA"],["JFK","BUF"]]), 
                  Input(tickets = [["HOU","JFK"],["SEA","JFK"],["JFK","SEA"],["JFK","HOU"]]), 
                  Input(tickets=[["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]),
                  Input(tickets=[["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]])]
        
        expected_outputs = [["JFK","BUF","HOU","SEA"], 
                            ["JFK","HOU","JFK","SEA","JFK"],
                            ["JFK","ATL","JFK","SFO","ATL","SFO"],
                            ["JFK","NRT","JFK","KUL"]]
        
        f = lambda x : sol.findItinerary(tickets=x.tickets)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_swimInWater(self):
        sol = Solution()
        
        class Input:
            def __init__(self, grid: List[List[int]]):
                self.grid = grid
                
        inputs = [Input(grid = [[0,1],[2,3]]), 
                  Input(grid = [[0,1,2,10],
                                [9,14,4,13],
                                [12,3,8,15],
                                [11,5,7,6]]),]
        
        expected_outputs = [3, 8]
        
        f = lambda x : sol.swimInWater(x.grid)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_foreignDictionary(self):
        sol = Solution()
        
        class Input:
            def __init__(self, words:List[str]):
                self.words = words
                
        inputs = [Input(words=["z","o"]), Input(words=["hrn","hrf","er","enn","rfnn"]), Input(words=["wrtkj","wrt"])]
        
        expected_outputs = ["zo", "hernf", ""]
        
        f = lambda x : sol.foreignDictionary(words=x.words)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_findCheapestFlights(self):
        sol = Solution()
        
        class Input:
            def __init__(self, n: int, flights: List[List[int]], src: int, dst: int, k: int):
                self.n = n
                self.flights = flights
                self.src = src
                self.dst = dst
                self.k = k
                
        inputs = [Input(n = 4, flights = [[0,1,200],[1,2,100],[1,3,300],[2,3,100]], src = 0, dst = 3, k = 1),
                  Input(n = 3, flights = [[1,0,100],[1,2,200],[0,2,100]], src = 1, dst = 2, k = 1)]
        
        expected_outputs = [500, 200]
        
        f = lambda x : sol.findCheapestPrice(x.n, x.flights, x.src, x.dst, x.k)
        
        self.results_helper(f, inputs, expected_outputs)
        
    
    def test_solveNQueens(self):
        sol = Solution()
        
        class Input:
            def __init__(self, n: int):
                self.n = n
                
        inputs = [Input(n=4), Input(n=1)]
        
        expected_outputs = [[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]], ["Q"]]
        
        f = lambda x : sol.solveNQueens(n=x.n)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_maxProfit(self):
        sol = Solution()
        
        class Input:
            def __init__(self, prices: List[int]):
                self.prices = prices
                
        inputs = [Input(prices = [1,3,4,0,4]), Input(prices = [1])]
        
        expected_outputs = [6, 0]
        
        f = lambda x : sol.maxProfit(prices=x.prices)
        
        self.results_helper(f, inputs, expected_outputs)