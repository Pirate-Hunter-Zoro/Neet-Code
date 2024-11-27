from typing import List
from data_structures.disjoint_set import UnionFind
from data_structures.heap import Heap
from data_structures.pair import Pair

class Solution:
    """Compilation of various solved problems
    """
    
    
    def mergeSort(self, pairs: List[Pair]) -> List[Pair]:
        """Implement Merge Sort.
        Merge Sort is a divide-and-conquer algorithm for sorting an array or list of elements. 
        It works by recursively dividing the unsorted list into n sub-lists, each containing one element. 
        Then, it repeatedly merges sub-lists to produce new sorted sub-lists until there is only one sub-list remaining.

        Objective:
        Given a list of key-value pairs, sort the list by key using Merge Sort. 
        If two key-value pairs have the same key, maintain their relative order in the sorted list.

        Args:
            pairs (List[Pair]): a list of key-value pairs, where each key-value has an integer key and a string value

        Returns:
            List[Pair]: the same list of pairs sorted by key value
        """

        if len(pairs) <= 1:
            return pairs
        else:
            left_pairs = self.mergeSort(pairs[:len(pairs)//2])
            right_pairs = self.mergeSort(pairs[len(pairs)//2:])
            new_pairs = []
            left_idx = 0
            right_idx = 0
            while left_idx < len(left_pairs) and right_idx < len(right_pairs):
                if left_pairs[left_idx].value <= right_pairs[right_idx].value:
                    new_pairs.append(left_pairs[left_idx])
                    left_idx += 1
                else:
                    new_pairs.append(right_pairs[right_idx])
                    right_idx += 1
            while left_idx < len(left_pairs):
                new_pairs.append(left_pairs[left_idx])
                left_idx += 1
            while right_idx < len(right_pairs):
                new_pairs.append(right_pairs[right_idx])
                right_idx += 1
            return new_pairs
        
        
    def topologicalSort(self, n: int, edges: List[List[int]]) -> List[int]:
        """Implement topological sort.
        Topological sort is an algorithm for linearly ordering the vertices of a directed acyclic graph such that for every directed edge (u,v) vertex u comes before v in the ordering.

        Given a directed graph, perform a topological sort on its vertices and return the order as a list of vertex labels. 
        There may be multiple valid topological sorts for a given graph, so you may return any valid ordering.

        If the graph contains a cycle, you should return an empty list to indicate that a topological sort is not possible.

        Args:
            n (int): number of vertices of graph
            edges (List[List[int]]): a list of pairs, each representing a directed edge in the form (u, v), where u is the source vertex and v is the destination vertex

        Returns:
            List[int]: list of vertices ordered topologically from the graph
        """

        generation = [0 for _ in range(n)]
        children = [set() for _ in range(n)]
        no_parent = set([i for i in range(n)])

        for edge in edges:
            u = edge[0]
            v = edge[1]
            children[u].add(v)
            if v in no_parent:
                no_parent.remove(v)

        def detect_cycle(node: int, seen: set[int]) -> bool:
            """Helper method to detect if a cycle is present in the graph
            """
            seen.add(node)
            for child in children[node]:
                if child in seen:
                    return True
                else:
                    seen_copy = set()
                    for i in seen:
                        seen_copy.add(i)
                    if detect_cycle(node=child, seen=seen_copy):
                        return True
            return False
            
        if any([detect_cycle(node=i, seen=set()) for i in range(n)]):
            return []

        def set_generation(node: int, value: int):
            """Helper method to propagate a generation update for a node and its descendents
            """
            generation[node] = value
            for child in children[node]:
                if generation[child] <= generation[node]:
                    set_generation(child, generation[node] + 1)
        
        for edge in edges:
            u = edge[0]
            v = edge[1]
            if generation[v] <= generation[u]:
                set_generation(node=v, value=generation[u]+1)
        
        node_pairs = [Pair(key=i, value=generation[i]) for i in range(n)]
        return [pair.key for pair in self.mergeSort(node_pairs)]
    
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of dictionary words.
        You are allowed to reuse words in the dictionary an unlimited number of times. You may assume all dictionary words are unique.

        Args:
            s (str): string we are attempting to construct
            wordDict (List[str]): word bank we are givent o construct s

        Returns:
            bool: whether or not it was possible to construct the word
        """
        wordSet = set([word for word in wordDict])
        # At index i, can s[0:i+1] be constructed from wordDict?
        sols = [False for _ in range(len(s))]
        sols[0] = s[0:1] in wordSet 
        for i in range(len(s)):
            for j in range(i):
                if sols[j] and (s[j+1:i+1] in wordSet):
                    sols[i] = True
                    break
            if s[:i+1] in wordSet:
                sols[i] = True
        
        return sols[len(s)-1]      
    
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """Given an array nums of unique integers, return all possible subsets of nums.
        The solution set must not contain duplicate subsets. 
        You may return the solution in any order.

        Args:
            nums (List[int]): list of numbers to find subsets of

        Returns:
            List[List[int]]: list of all possible subsets of nums
        """
        # DP setup
        nums.sort()
        subsets_by_size = {}
        for size in range(len(nums)+1):
            subsets_by_size[size] = [[] for _ in range(len(nums))]
        
        # Base case - there is one subset of size 0 - the empty set
        for lowest_index_included in range(0, len(nums)):
            subsets_by_size[1][lowest_index_included].append([nums[lowest_index_included]])
        
        # Bottom up dynamic programming
        subsets = [[]] + [[n] for n in nums]
        for size in range(2, len(nums)+1):
            for lowest_index_included in range(0, len(nums)+1-size):
                for next_lowest_index_included in range(lowest_index_included+1, len(nums)+1-(size-1)):
                    for lower_size_subset in subsets_by_size[size-1][next_lowest_index_included]:
                        new_subset = [nums[lowest_index_included]] + lower_size_subset
                        subsets.append(new_subset)
                        subsets_by_size[size][lowest_index_included].append(new_subset)
                    
        return subsets
    
    
    def canPartition(self, nums: List[int]) -> bool:
        """You are given an array of positive integers nums.
        Return true if you can partition the array into two subsets, subset1 and subset2 where sum(subset1) == sum(subset2). 
        Otherwise, return false.

        Args:
            nums (List[int]): array of positive integers to partition

        Returns:
            bool: whether the array can be partitioned as described
        """
        nums.sort()
        total = sum(nums)
        if total % 2 == 1:
            return False
        
        # Otherwise, divide the sum by 2 and achieve knapsack
        target_sum = total // 2
        sols = [[False for _ in nums] for _ in range(target_sum)]
        for target in range(1, target_sum+1):
            for idx, num in enumerate(nums):
                if num == target:
                    sols[target-1][idx] = True
                else:
                    if num < target and idx > 0:
                        # Try picking
                        sols[target-1][idx] = sols[target - 1 - num][idx-1]
                    if idx > 0:
                        sols[target-1][idx] = sols[target - 1][idx] or sols[target - 1][idx-1]
            
        # Return our final solution
        return sols[target_sum-1][len(nums)-1]
    
    
    def lengthOfLIS(self, nums: List[int]) -> int:
        """Given an integer array nums, return the length of the longest strictly increasing subsequence.
        A subsequence is a sequence that can be derived from the given sequence by deleting some or no elements without changing the relative order of the remaining characters.

        Args:
            nums (List[int]): list of numbers to find length of LIS of

        Returns:
            int: length of the list's LIS
        """
        running_sequence = [nums[0]]
        def findLowestAbove(num: int) -> int:
            left = 0
            right = len(running_sequence)
            while left < right:
                mid = (left + right) // 2
                if running_sequence[mid] > num:
                    if running_sequence[mid-1] == num:
                        return mid-1
                    elif running_sequence[mid-1] < num:
                        return mid
                    else:
                        # Look left
                        right = mid
                elif running_sequence[mid] == num:
                    return mid
                else:
                    # Look right
                    left = mid + 1
            return left
        
        for num in nums[1:]:
            if running_sequence[len(running_sequence)-1] < num:
                running_sequence.append(num)
            else:
                # Find the lowest number which does not fall below this number
                running_sequence[findLowestAbove(num)] = num
        
        return len(running_sequence)
    
    
    def numDistinct(self, s: str, t: str) -> int:
        """You are given two strings s and t, both consisting of english letters.
        Return the number of distinct subsequences of s which are equal to t.

        Args:
            s (str): string whose subsequences that equal t we are counting
            t (str): target string for the subsequences of s to equal

        Returns:
            int: number of subsequences of s which equal t
        """
        sols = [[0 for _ in range(len(s))] for _ in range(len(t))]
        
        # Base case - consider matching the first character of t from any character in s from range 0-j
        if s[0] == t[0]:
            sols[0][0] = 1
        for j in range(1, len(s)):
            if s[j] == t[0]:
                sols[0][j] = 1 + sols[0][j-1]
            else:
                sols[0][j] = sols[0][j-1]
                
        # Now we build up via dynamic programming
        for i in range(1, len(t)):
            # How many ways to match t[:i+1] with subsequences in s[:j+1]?
            for j in range(i, len(s)):
                if t[i] == s[j]:
                    # Try matching
                    sols[i][j] = sols[i-1][j-1]
                    # Try not matching
                    sols[i][j] += sols[i][j-1]
                else:
                    # No choice but to not match
                    sols[i][j] = sols[i][j-1]
        
        return sols[len(t)-1][len(s)-1]
    
    
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        """You are given a 2D integer array intervals, where intervals[i] = [left_i, right_i] represents the ith interval starting at left_i and ending at right_i (inclusive).
        You are also given an integer array of query points queries. 
        The result of query[j] is the length of the shortest interval i such that left_i <= queries[j] <= right_i. 
        If no such interval exists, the result of this query is -1.

        Return an array output where output[j] is the result of query[j].

        Note: The length of an interval is calculated as right_i - left_i + 1.

        Args:
            intervals (List[List[int]]): list of intervals
            queries (List[int]): list of queries

        Returns:
            List[int]: query results
        """
        results = [-1 for _ in queries]
        
        # We will need a heap for this, so we will use the following comparator
        def compare(interval_1: list[int], interval_2: list[int]) -> bool:
            """Helper function to compare two intervals

            Args:
                interval_1 (list[int]): first interval start and end times
                interval_2 (list[int]): second interval start and end times

            Returns:
                int: result representing if the first interval should precede the second interval in a heap
            """
            first_length = interval_1[1] - interval_1[0]
            second_length = interval_2[1] - interval_2[0]
            first_end = interval_1[1]
            second_end = interval_2[1]
            if first_length < second_length:
                return True
            elif first_length > second_length:
                return False
            else:
                return first_end < second_end
        interval_heap = Heap(comparator=compare)
            
        # We now sort our intervals by starting time
        intervals.sort(key=lambda interval: interval[0])
        
        # And sort our queries, but for purposes of the output remember original order
        query_inidices = {}
        for i, q in enumerate(queries):
            if q not in query_inidices.keys():
                query_inidices[q] = [i]
            else:
                query_inidices[q].append(i)
        queries.sort()
        
        current_interval_idx = 0
        query_value_results = {} # Map query value to respective index
        # Now for each query, add all NEW intervals which have a starting time less than or equal to the 
        for query in queries:
            while current_interval_idx < len(intervals):
                current_interval = intervals[current_interval_idx]
                start = current_interval[0]
                end = current_interval[1]
                if start <= query and end >= query:
                    # The query will fit in this interval
                    interval_heap.push(val=current_interval)
                    current_interval_idx += 1
                elif start <= query:
                    # Useless interval - won't fit in this query or any later query
                    current_interval_idx += 1
                else:
                    # Won't fit with this query, but maybe with a future query
                    break
            # Find the first interval that works with this query, and don't pop it but pop all preceding
            # If we run out of intervals, the result for this query is -1
            while not interval_heap.empty():
                next_interval = interval_heap.top()
                if next_interval[0] <= query and next_interval[1] >= query:
                    # This next smallest interval can accomodate the query - record the length of this interval for this query
                    query_value_results[query] = next_interval[1] - next_interval[0] + 1
                    break
                else:
                    # This interval was too early to fit the current query, and it won't fit any later queries
                    interval_heap.pop()
                    
        # Now that we have our query results, we need to put the results back in the original order that the queries came in
        for query, result in query_value_results.items():
            for query_idx in query_inidices[query]:
                results[query_idx] = result
            
        return results
    
    
    def minimumSpanningTree(self, n: int, edges: List[List[int]]) -> int:
        """Implement Kruskal's minimum spanning tree algorithm.

        A Minimum Spanning Tree (MST) is a tree that spans all the vertices in a given weighted, undirected graph while minimizing the total edge weight and avoiding cycles. 
        It connects all nodes with exactly ∣V∣−1 edges, where V is the set of vertices, and has the lowest possible sum of edge weights.

        Kruskal's algorithm is a greedy algorithm that finds the MST of graph. 
        It sorts all the edges from least weight to greatest, and iteratively adds edges to the MST, ensuring that each new edge doesn't form a cycle.

        Objective:

        Given a weighted, undirected graph, find the minimum spanning tree (MST) using Kruskal's algorithm and return its total weight. 
        If the graph is not connected, the total weight of the minimum spanning tree should be -1.

        Input:

        n - the number of vertices in the graph, where (2 <= n <= 100). Each vertex is labeled from 0 to n - 1.
        edges - a list of tuples, each representing an undirected edge in the form (u, v, w), where u and v are vertices connected by the edge, and w is the weight of the edge, where (1 <= w <= 10).
        Note: If the graph is not connected, you should return -1.

        Args:
            n (int): number of nodes
            edges (List[List[int]]): edges of the graph

        Returns:
            int: weight of the minimum spanning tree
        """
        def compare_edges(first_edge: List[int], second_edge: List[int]) -> bool:
            """Helper method to determine if the first edge is less than the second

            Args:
                first_edge (List[int]): first edge
                second_edge (List[int]): second edge

            Returns:
                bool: whether the first edge is less than the second edge
            """
            return first_edge[2] < second_edge[2]
        edge_heap = Heap(comparator=compare_edges)
        for edge in edges:
            edge_heap.push(edge)
        
        used_edges = 0
        weight = 0
        node_set = UnionFind(n=n)
        while used_edges < n-1 and not edge_heap.empty():
            next_edge = edge_heap.pop()
            node_1 = next_edge[0]
            node_2 = next_edge[1]
            if not node_set.isSameComponent(x=node_1, y=node_2):
                node_set.union(x=node_1, y=node_2)
                weight += next_edge[2]
                used_edges += 1
        
        return weight if used_edges == n-1 else -1