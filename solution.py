from typing import List
from pair import Pair

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
        pass