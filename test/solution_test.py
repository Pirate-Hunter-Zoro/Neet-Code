from typing import List, Optional
import unittest

from implementation.linked_list import ListNode
from implementation.tree_node import TreeNode
from implementation.solution import Solution

from typing import List

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
        
        expected_outputs = [[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]], [["Q"]]]
        
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
        
    
    def test_minDistance(self):
        sol = Solution()
        
        class Input:
            def __init__(self, word1: str, word2: str):
                self.word1 = word1
                self.word2 = word2
        
        inputs = [Input(word1 = "monkeys", word2 = "money"), Input(word1 = "neatcdee", word2 = "neetcode")]
        
        expected_outputs = [2, 3]
        
        f = lambda x : sol.minDistance(word1=x.word1, word2=x.word2)
        
        self.results_helper(f, inputs, expected_outputs)
        
    
    def test_uniquePaths(self):
        sol = Solution()
        
        class Input:
            def __init__(self, m: int, n: int):
                self.m = m
                self.n = n
                
        inputs = [Input(m = 3, n = 6), Input(m = 3, n = 3)]
        
        expected_outputs = [21, 6]
        
        f = lambda x : sol.uniquePaths(m=x.m, n=x.n)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_combinationSum(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int], target: int):
                self.nums = nums
                self.target = target
        
        inputs = [Input(nums = [2,5,6,9], target=9), Input(nums = [3,4,5], target=16), Input(nums=[3], target=5), Input(nums=[7,3,2], target=18)]
        
        expected_outputs = [[[2,2,5],[9]], [[3,3,3,3,4],[3,3,5,5],[3,4,4,5],[4,4,4,4]], [], [[7,7,2,2],[7,3,3,3,2],[7,3,2,2,2,2],[3,3,3,3,3,3],[3,3,3,3,2,2,2],[3,3,2,2,2,2,2,2],[2,2,2,2,2,2,2,2,2]]]
        
        f = lambda x : sol.combinationSum(nums=x.nums, target=x.target)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_combinationSum2(self):
        sol = Solution()
        
        class Input:
            def __init__(self, candidates: List[int], target: int):
                self.candidates = candidates
                self.target = target
                
        inputs = [Input(candidates = [9,2,2,4,6,1,5], target = 8), Input(candidates = [1,2,3,4,5], target = 7)]
        
        expected_outputs = [[
                            [2,6],
                            [1,2,5],
                            [2,2,4],
                            ], [
                            [2,5],
                            [3,4],
                            [1,2,4],
                            ]]
        
        f = lambda x : sol.combinationSum2(candidates=x.candidates, target=x.target)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_permute(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int]):
                self.nums = nums
                
        inputs = [Input(nums = [1,2,3]), Input(nums = [7])]
        
        expected_outputs = [[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]], [[7]]]
        
        f = lambda x : sol.permute(x.nums)
        
        self.results_helper(f, inputs, expected_outputs)
        
    
    def test_subsetsWithDup(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums: List[int]):
                self.nums = nums
        
        inputs = [Input(nums = [1,2,1]), Input(nums = [7,7])]
        
        expected_outputs = [[[1,1,2], [1,1], [1,2], [1], [2], []], [[7,7], [7], []]]
        
        f = lambda x : sol.subsetsWithDup(nums=x.nums)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_mergeKLists(self):
        sol = Solution()
        
        class Input:
            def __init__(self, lists: List[ListNode]):
                self.lists = lists
        
        inputs = [Input(lists = [ListNode(values=[1,2,4]),ListNode(values=[1,3,5]),ListNode(values=[3,6])]), Input(lists=[]), Input(lists=[ListNode(values=[])])]
        
        expected_outputs = [ListNode(values=[1,1,2,3,3,4,5,6]), None, None]
        
        f = lambda x : sol.mergeKLists(lists=x.lists)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_exist(self) -> bool:
        sol = Solution()
        
        class Input:
            def __init__(self, board: List[List[str]], word: str):
                self.board = board
                self.word = word
                
        inputs = [Input(board = [
                                ["A","B","C","D"],
                                ["S","A","A","T"],
                                ["A","C","A","E"]
                                ],
                                word = "CAT"),
                  Input(board = [
                                ["A","B","C","D"],
                                ["S","A","A","T"],
                                ["A","C","A","E"]
                                ],
                                word = "BAT")]
        
        expected_outputs = [True, False]
        
        f = lambda x : sol.exist(x.board, x.word)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_findWords(self):
        sol = Solution()
        
        class Input:
            def __init__(self, board: List[List[str]], words: List[str]):
                self.board = board
                self.words = words
        
        inputs = [Input(board = [
                                ["a","b","c","d"],
                                ["s","a","a","t"],
                                ["a","c","k","e"],
                                ["a","c","d","n"]
                                ],
                                words = ["bat","cat","back","backend","stack"]),
                    Input(
                        board = [
                                ["x","o"],
                                ["x","o"]
                                ],
                                words = ["xoxo"]
                  )]
        
        expected_outputs = [["back","backend","cat"], []]
        
        f = lambda x : sol.findWords(board=x.board, words=x.words)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_partition(self):
        sol = Solution()
        
        class Input:
            def __init__(self, s: str):
                self.s = s
                
        inputs = [Input(s = "aab"), Input(s = "a")]
        
        expected_outputs = [[["a","a","b"],["aa","b"]], [["a"]]]
        
        f = lambda x : sol.partition(s=x.s)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_letterCombinations(self):
        sol = Solution()
        
        class Input:
            def __init__(self, digits:str):
                self.digits = digits
        
        inputs = [Input(digits = "34"), Input(digits = "")]
        
        expected_outputs = [["dg","dh","di","eg","eh","ei","fg","fh","fi"], []]
        
        f = lambda x : sol.letterCombinations(digits=x.digits)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_findMedianSortedArrays(self):
        sol = Solution()
        
        class Input:
            def __init__(self, nums1: List[int], nums2: List[int]):
                self.nums1 = nums1
                self.nums2 = nums2
                
        inputs = [Input(nums1 = [1,2], nums2 = [3]), Input(nums1 = [1,3], nums2 = [2,4])]
        
        expected_outputs = [2.0, 2.5]
        
        f = lambda x : sol.findMedianSortedArrays(nums1=x.nums1, nums2=x.nums2)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_trap(self):
        sol = Solution()
        
        class Input:
            def __init__(self, height: List[int]):
                self.height = height
                
        inputs = [Input(height = [0,2,0,3,1,0,1,3,2,1])]
        
        expected_outputs = [9]
        
        f = lambda x : sol.trap(height=x.height)
        
        self.results_helper(f, inputs, expected_outputs)
        

    def test_largestRectangleArea(self):
        sol = Solution()
        
        class Input:
            def __init__(self, heights: List[int]):
                self.heights = heights
        
        inputs = [Input(heights = [7,1,7,2,2,4]), Input(heights = [1,3,7])]
        
        expected_outputs = [8,7]
        
        f = lambda x : sol.largestRectangleArea(heights=x.heights)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_longestIncreasingPath(self):
        sol = Solution()
        
        class Input:
            def __init__(self, matrix: List[List[int]]):
                self.matrix = matrix
                
        inputs = [Input(matrix = [[5,5,3],[2,3,6],[1,1,1]]), Input(matrix = [[1,2,3],[2,1,4],[7,6,5]])]
        
        expected_outputs = [4, 7]
        
        f = lambda x : sol.longestIncreasingPath(matrix=x.matrix)
        
        self.results_helper(f, inputs, expected_outputs)
        
        
    def test_isMatch(self):
        sol = Solution()
        
        class Input:
            def __init__(self, s: str, p: str):
                self.s = s
                self.p = p
                
        inputs = [Input(s = "aa", p = ".b"), Input(s = "nnn", p = "n*"), Input(s = "xyz", p = ".*z"), Input(s="aaa", p="ab*ac*a")]
        
        expected_outputs = [False, True, True, True]
        
        f = lambda x : sol.isMatch(s=x.s, p=x.p)
        
        self.results_helper(f, inputs, expected_outputs)