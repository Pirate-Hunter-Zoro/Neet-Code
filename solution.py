from typing import Dict, List, Optional, Self
from data_structures.disjoint_set import UnionFind
from data_structures.heap import Heap
from data_structures.pair import Pair
from data_structures.tree_node import TreeNode
from data_structures.linked_list import Queue

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
    
    
    def minimumSpanningTreeKruskal(self, n: int, edges: List[List[int]]) -> int:
        """Implement Kruskal's minimum spanning tree algorithm.

        A Minimum Spanning Tree (MST) is a tree that spans all the vertices in a given weighted, undirected graph while minimizing the total edge weight and avoiding cycles. 
        It connects all nodes with exactly ∣V∣−1 edges, where V is the set of vertices, and has the lowest possible sum of edge weights.

        Kruskal's algorithm is a greedy algorithm that finds the MST of graph. 
        It sorts all the edges from least weight to greatest, and iteratively adds edges to the MST, ensuring that each new edge doesn't form a cycle.

        Objective:

        Given a weighted, undirected graph, find the minimum spanning tree (MST) using Kruskal's algorithm and return its total weight. 
        If the graph is not connected, the total weight of the minimum spanning tree should be -1.

        Input:

        n - the number of vertices in the graph, where (2 <= n <= 100). 
            Each vertex is labeled from 0 to n - 1.
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
    
    def minimumSpanningTreePrim(self, n: int, edges: List[List[int]]) -> int:
        """Implement Prim's minimum spanning tree algorithm.

        A Minimum Spanning Tree (MST) is a tree that spans all the vertices in a given weighted, undirected graph while minimizing the total edge weight and avoiding cycles. It connects all nodes with exactly ∣V∣−1 edges, where V is the set of vertices, and has the lowest possible sum of edge weights.

        Prim's algorithm is a greedy algorithm that builds the MST of a graph starting from an arbitrary vertex. 
        At each step, the algorithm adds the lightest edge connecting a vertex in the MST to a vertex outside the MST, effectively "growing" the MST one edge at a time.

        Objective:

        Given a weighted, undirected graph, find the minimum spanning tree (MST) using Prim's algorithm and return its total weight. 
        If the graph is not connected, the total weight of the minimum spanning tree should be -1.

        Input:

        n - the number of vertices in the graph, where (2 <= n <= 100). 
            Each vertex is labeled from 0 to n - 1.
        edges - a list of tuples, each representing an undirected edge in the form (u, v, w), where u and v are vertices connected by the edge, and w is the weight of the edge, where (1 <= w <= 10).

        Args:
            n (int): number of nodes
            edges (List[List[int]]): list of edges

        Returns:
            int: weight of minimum spanning tree
        """
        adjacency_list = [[] for _ in range(n)]
        for edge in edges:
            a = edge[0]
            b = edge[1]
            w = edge[2]
            adjacency_list[a].append([b, w])
            adjacency_list[b].append([a, w])
            
        def less(edge_1: List[int], edge_2: List[int]) -> bool:
            """Compare two edges by weight

            Args:
                edge_1 (List[int]): first edge
                edge_2 (List[int]): second edge

            Returns:
                bool: whether the first edge is less than the second edge
            """
            return edge_1[1] < edge_2[1]
        edge_heap = Heap(comparator=less)
            
        seen_nodes = set([0])
        for edge in adjacency_list[0]:
            if edge[0] != 0: # no self loops allowed
                edge_heap.push(edge)
        weight = 0
        while len(seen_nodes) < n and not edge_heap.empty():
            # Repeatedly expand from the nodes we have seen thus far
            next_edge = edge_heap.pop()
            next_node = next_edge[0]
            next_weight = next_edge[1]
            if next_node not in seen_nodes:
                seen_nodes.add(next_node)
                weight += next_weight
                for edge in adjacency_list[next_node]:
                    if edge[0] not in seen_nodes:
                        edge_heap.push(edge)
            
        if len(seen_nodes) < n:
            return -1
        else:
            return weight
        
    def shortestPath(self, n: int, edges: List[List[int]], src: int) -> Dict[int, int]:
        """Implement Dijkstra's shortest path algorithm.

        Given a weighted, directed graph, and a starting vertex, return the shortest distance from the starting vertex to every vertex in the graph.

        Input:

        n - the number of vertices in the graph, where (2 <= n <= 100). Each vertex is labeled from 0 to n - 1.
        edges - a list of tuples, each representing a directed edge in the form (u, v, w), where u is the source vertex, v is the destination vertex, and w is the weight of the edge, where (1 <= w <= 10).
        src - the source vertex from which to start the algorithm, where (0 <= src < n).
        Note: If a vertex is unreachable from the source vertex, the shortest path distance for the unreachable vertex should be -1.

        Args:
            n (int): number of nodes
            edges (List[List[int]]): list of edges
            src (int): source node

        Returns:
            Dict[int, int]: distance to all other vertices
        """
        distances = {}
        distances[src] = 0
        for i in range(0,n):
            if i != src:
                distances[i] = -1
        
        # We need edges
        adjacency_list = [[] for _ in range(n)]
        for edge in edges:
            a = edge[0]
            b = edge[1]
            w = edge[2]
            adjacency_list[a].append([b, w])
            
        # We need a heap of distances
        def less(distance_1: List[int], distance_2: List[int]) -> bool:
            """Compare two edges by weight

            Args:
                distance_1 (List[int]): first distance (total distance from node 0, this node)
                distance_2 (List[int]): second edge (total distance from node 0, this node)

            Returns:
                bool: whether the first edge is less than the second edge
            """
            return distance_1[1] < distance_2[1]
        distance_heap = Heap(comparator=less)
        
        # The round-1 iteration distances from src to all its neighbors is simply the edge weight
        for edge in adjacency_list[src]:
            distance_heap.push(edge)
        
        while not distance_heap.empty():
            next_distance = distance_heap.pop()
            node = next_distance[0]
            new_distance_from_0 = next_distance[1]
            old_distance_from_0 = distances[node]
            if distances[node] != -1:
                distances[node] = min(distances[node], new_distance_from_0)
            else:
                distances[node] = new_distance_from_0
            if distances[node] != old_distance_from_0:
                for edge in adjacency_list[node]:
                    next_node = edge[0]
                    weight = edge[1]
                    distance_heap.push([next_node, distances[node] + weight])

        return distances
    
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """Given the root of a non-empty binary tree, return the maximum path sum of any non-empty path.
        A path in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge connecting them. 
        A node can not appear in the sequence more than once. 
        The path does not necessarily need to include the root.

        The path sum of a path is the sum of the node's values in the path.

        Args:
            root (Optional[TreeNode]): the tree node (which may be null)

        Returns:
            int: maximum path sum in the tree
        """
        def recMaxPathSum(root: Optional[TreeNode]) -> tuple[int, int]:
            """Return three things
            - Result from including this root node and ONLY its best subtree (at most)
            - Result from taking the true best path in the entire subtree of this node

            Args:
                root (Optional[TreeNode]): node we are talking about

            Returns:
                tuple[int, int, int]: tuple of said three results
            """
            if root.left == None and root.right == None:
                # root node
                return (root.val, root.val)
            else:
                if root.right == None:
                    # only left child
                    best_include_left, best_left = recMaxPathSum(root.left)
                    # now we can fill in this node's values
                    return (root.val + max(0, best_include_left), max(best_left, root.val + max(0, best_include_left)))
                elif root.left == None:
                    # only right child
                    best_include_right, best_right = recMaxPathSum(root.right)
                    # now we can fill in this node's values
                    return (root.val + max(0, best_include_right), max(best_right, root.val + max(0, best_include_right)))
                else:
                    # both children
                    best_include_left, best_left = recMaxPathSum(root.left)
                    best_include_right, best_right = recMaxPathSum(root.right)
                    # now we solve for this node
                    # if we enforce including this node and at most one of its subtrees
                    include_with_subtree = root.val + max(0, max(best_include_left, best_include_right))
                    # or we need not include either subtree or even this root node, OR both subtrees
                    include_root = root.val + max(0, max(best_include_left + best_include_right, max(best_include_left, best_include_right)))
                    no_include_root = max(best_left, best_right)
                    best_path = max(include_root, no_include_root)
                    return (include_with_subtree, best_path)
        _, best = recMaxPathSum(root)
        return best
    
    def maxCoins(self, nums: List[int]) -> int:
        """You are given an array of integers nums of size n. 
        The ith element represents a balloon with an integer value of nums[i]. 
        You must burst all of the balloons.
        If you burst the ith balloon, you will receive nums[i - 1] * nums[i] * nums[i + 1] coins. 
        If i - 1 or i + 1 goes out of bounds of the array, then assume the out of bounds value is 1.
        
        Return the maximum number of coins you can receive by bursting all of the balloons.

        Args:
            nums (List[int]): balloon values

        Returns:
            int: maximum number of coins that can be earned
        """
        # We need to make the decision - what's the LAST balloon to pop?
        # [b1, b2, b3, b4, ..., bn]
        # Subproblem defined as (left-end, right-end, start-idx, end-idx)
        sols = {}
        def topDownMaxCoins(left_val: int, right_val: int, start_idx: int, end_idx: int) -> int:
            """Helper method to solve the burst balloons problem

            Args:
                left_val (int): value of the balloon on the left side of our range of balloons we will pop
                right_val (int): value of the balloon on the right side of our range of balloons we will pop
                start_idx (int): left index of range of balloons to pop
                end_idx (int): right index of range of balloons to pop

            Returns:
                int: maximum coins we can achieve if we pop the range of balloons optimally with the given left and right-side values
            """
            # See if we need to solve the problem or if we have already solved it
            if left_val not in sols.keys():
                sols[left_val] = {}
            if right_val not in sols[left_val].keys():
                sols[left_val][right_val] = {}
            if start_idx not in sols[left_val][right_val].keys():
                sols[left_val][right_val][start_idx] = {}
            if end_idx not in sols[left_val][right_val][start_idx].keys():
                # Here is where we will solve the problem
                if end_idx == start_idx:
                    # Only one balloon - base case
                    sols[left_val][right_val][start_idx][end_idx] = nums[start_idx] * left_val * right_val
                else:
                    # Not a base case
                    record = -1
                    for i in range(start_idx, end_idx+1):
                        # See what the best balloon to leave last for popping is
                        result = left_val*nums[i]*right_val
                        if i > start_idx:
                            # Optimally pop the left
                            result += topDownMaxCoins(left_val=left_val, right_val=nums[i], start_idx=start_idx, end_idx=i-1)
                        if i < end_idx:
                            # Optimally pop the right
                            result += topDownMaxCoins(left_val=nums[i], right_val=right_val, start_idx=i+1, end_idx=end_idx)
                        record = max(record, result)
                    sols[left_val][right_val][start_idx][end_idx] = record
            
            return sols[left_val][right_val][start_idx][end_idx]
        
        return topDownMaxCoins(1, 1, 0, len(nums)-1)
    
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """You are given two words, beginWord and endWord, and also a list of words wordList. 
        All of the given words are of the same length, consisting of lowercase English letters, and are all distinct.
        
        Your goal is to transform beginWord into endWord by following the rules:
        - You may transform beginWord to any word within wordList, provided that at exactly one position the words have a different character, and the rest of the positions have the same characters.
        - You may repeat the previous step with the new word that you obtain, and you may do this as many times as needed.
        - Return the minimum number of words within the transformation sequence needed to obtain the endWord, or 0 if no such sequence exists.

        Args:
            beginWord (str): starting word
            endWord (str): word to transform the starting word into
            wordList (List[str]): list of words to change into to reach the end word

        Returns:
            int: minimum number of words needed for transformation
        """
        class WordNode:
            def __init__(self, word: str):
                self.word = word
                self.visited = False
                self.connections = []
                self.depth = 1
                
        graph = {}
        words = wordList
        words.append(beginWord)
        # Note that because our graph is a hash map, any duplicates in words will not result in duplicate nodes
        for word in words:
            if word not in graph.keys():
                graph[word] = WordNode(word)
        
        unique_words = [word for word in graph.keys()]
        # Now look at every possible pair and see if they are connected in our underlying graph
        for i in range(len(unique_words)):
            word_1 = unique_words[i]
            for j in range(i+1, len(unique_words)):
                word_2 = unique_words[j]
                differences = sum([1 if word_1[i] != word_2[i] else 0 for i in range(len(word_1))])
                if differences == 1:
                    graph[word_1].connections.append(graph[word_2])
                    graph[word_2].connections.append(graph[word_1])
        
        # Now we perform breadth-first-search
        node_queue = Queue()
        node_queue.push(graph[beginWord])
        while len(node_queue) > 0:
            next = node_queue.pop()
            if next.word == endWord:
                return next.depth
            else:
                for word_node in next.connections:
                    if not word_node.visited:
                        word_node.visited = True
                        word_node.depth = next.depth + 1
                        node_queue.push(word_node)
                        
        return 0
    
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        """You are given a list of flight tickets tickets where tickets[i] = [from_i, to_i] represent the source airport and the destination airport.
        Each from_i and to_i consists of three uppercase English letters.
        Reconstruct the itinerary in order and return it.
        All of the tickets belong to someone who originally departed from "JFK". 
        Your objective is to reconstruct the flight path that this person took, assuming each ticket was used exactly once.
        If there are multiple valid flight paths, return the lexicographically smallest one.
        For example, the itinerary ["JFK", "SEA"] has a smaller lexical order than ["JFK", "SFO"].
        You may assume all the tickets form at least one valid flight path.

        Args:
            tickets (List[List[str]]): list of flight tickets

        Returns:
            List[str]: sequence of visited cities
        """
        adjacency_list = {}
        # Each node gets a heap of connections (going by alphabetical order)
        for ticket in tickets:
            src = ticket[0]
            dst = ticket[1]
            if src not in adjacency_list.keys():
                adjacency_list[src] = []
            adjacency_list[src].append(dst)
        for neighbors in adjacency_list.values():
            neighbors.sort()
                
        def recFindItinerary(current: str, num_edges_used: int, current_itinerary: list[str]):
            """Helper method to recursively build up the itinerary from the current node

            Args:
                current (str): current city
                num_edges_used (int): number of edges used so far
                current_itinerary: list of cities progressed so far

            Returns:
                list[str]: resulting itinerary from this point
            """
            current_itinerary.append(current)
            if num_edges_used < len(tickets):
                # Non base case
                successful = False
                if current in adjacency_list.keys():
                    # We have neighbors to explore
                    for i in range(len(adjacency_list[current])):
                        neighbor = adjacency_list[current][i]
                        adjacency_list[current].remove(neighbor)
                        recFindItinerary(current=neighbor, num_edges_used=num_edges_used+1, current_itinerary=current_itinerary)
                        successful = len(current_itinerary) == len(tickets)+1
                        if successful:
                            break
                        else:
                            adjacency_list[current].insert(i, neighbor)
                if not successful:
                    # Remove this current node from the itinerary
                    current_itinerary.pop()
        
        itinerary = []
        recFindItinerary(current="JFK", num_edges_used=0, current_itinerary=itinerary)
        return itinerary
    
    def swimInWater(self, grid: List[List[int]]) -> int:
        """You are given a square 2-D matrix of distinct integers grid where each integer grid[i][j] represents the elevation at position (i, j).
        Rain starts to fall at time = 0, which causes the water level to rise. 
        At time t, the water level across the entire grid is t.
        You may swim either horizontally or vertically in the grid between two adjacent squares if the original elevation of both squares is less than or equal to the water level at time t.
        Starting from the top left square (0, 0), return the minimum amount of time it will take until it is possible to reach the bottom right square (n - 1, n - 1).

        Args:
            grid (List[List[int]]): elevated position map

        Returns:
            int: minimum time it will take to reach the bottom right square
        """
        # This appears to be a modified djikstra's algorithm where the weight of the path is the maximum along the path rather than the sum of edge weights
        class GridSpace:
            def __init__(self, row: int, col: int, wait_time: int):
                self.row = row
                self.col = col
                self.wait_time = wait_time
                self.path_wait_time = 1000000
                                
            def __lt__(self, other: Self):
                return self.path_wait_time < other.path_wait_time
            
            def __gt__(self, other: Self):
                return self.path_wait_time > other.path_wait_time
        
        spaces = [[GridSpace(row=i, col=j, wait_time=grid[i][j]) for j in range(len(grid[0]))] for i in range(len(grid))]
        spaces[0][0].path_wait_time = spaces[0][0].wait_time
        
        grid_space_heap = Heap()
        grid_space_heap.push(spaces[0][0])
        while not grid_space_heap.empty():
            next = grid_space_heap.pop()
            if next.row == len(grid)-1 and next.col == len(grid)-1:
                break
            else:
                neighbors = []
                if next.row > 0:
                    neighbors.append(spaces[next.row-1][next.col])
                if next.row < len(grid)-1:
                    neighbors.append(spaces[next.row+1][next.col])
                if next.col > 0:
                    neighbors.append(spaces[next.row][next.col-1])
                if next.col < len(grid)-1:
                    neighbors.append(spaces[next.row][next.col+1])
                
                for neighbor in neighbors:
                    new_neighbor_path_time = min(neighbor.path_wait_time, max(next.path_wait_time, neighbor.wait_time))
                    if new_neighbor_path_time < neighbor.path_wait_time:
                        # The neighbor will get updated - add them to the heap
                        neighbor.path_wait_time = new_neighbor_path_time
                        grid_space_heap.push(neighbor)
        
        return spaces[len(spaces)-1][len(spaces)-1].path_wait_time
    
    def foreignDictionary(self, words: List[str]) -> str:
        """There is a foreign language which uses the latin alphabet, but the order among letters is not "a", "b", "c" ... "z" as in English.
        You receive a list of non-empty strings words from the dictionary, where the words are sorted lexicographically based on the rules of this new language.
        Derive the order of letters in this language. 
        If the order is invalid, return an empty string. 
        If there are multiple valid order of letters, return any of them.
        A string a is lexicographically smaller than a string b if either of the following is true:
        - The first letter where they differ is smaller in a than in b.
        - There is no index i such that a[i] != b[i] and a.length < b.length.

        Args:
            words (List[str]): list of words sorted according to new rules

        Returns:
            str: string describing the order of letters
        """
        # First we will map each character to a digit, and vice versa
        char_to_int = {}
        int_to_char = {}
        for word in words:
            for i in range(len(word)):
                char = word[i]
                if char not in char_to_int.keys():
                    char_to_int[char] = len(char_to_int)
                    int_to_char[len(char_to_int)-1] = char
                    
        connections = []   
        invalid = [False]
        def make_connections(start: int, end: int, idx: int):
            chars_at_level = []
            char_ranges = {}
            last_char = None
            for i in range(start, end):
                word = words[i]
                if idx < len(word):
                    char = word[idx]
                    if len(chars_at_level) == 0 or chars_at_level[len(chars_at_level)-1] != char:
                        chars_at_level.append(char)
                        char_ranges[char] = [i]
                        if i > start and last_char != None:
                            char_ranges[last_char].append(i)
                        last_char = char
                    if i == end-1:
                        char_ranges[char].append(end)
                else:
                    if last_char != None:
                        invalid[0] = True
            
            for index_range in char_ranges.values():
                if len(index_range) == 2:
                    make_connections(start=index_range[0], end=index_range[1], idx=idx+1)
                
            for i in range(len(chars_at_level)-1, 0, -1):
                char = chars_at_level[i]
                lower_char = chars_at_level[i-1]
                connections.append([char_to_int[char], char_to_int[lower_char]])
                
        make_connections(start=0, end=len(words), idx=0)
        if invalid[0]:
            return ""
        
        result = ""
        reverse_alphabet = self.topologicalSort(n=len(char_to_int), edges=connections)
        # Now reverse the reverse alphabet
        for i in range(len(reverse_alphabet)-1, -1, -1):
            result += int_to_char[reverse_alphabet[i]]
        
        return result
    
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """There are n airports, labeled from 0 to n - 1, which are connected by some flights. 
        You are given an array flights where flights[i] = [from_i, to_i, price_i] represents a one-way flight from airport from_i to airport to_i with cost price_i. 
        You may assume there are no duplicate flights and no flights from an airport to itself.
        
        You are also given three integers src, dst, and k where:
        - src is the starting airport
        - dst is the destination airport
        - src != dst
        - k is the maximum number of stops you can make (not including src and dst)
        
        Return the cheapest price from src to dst with at most k stops, or return -1 if it is impossible.

        Args:
            n (int): number of airports
            flights (List[List[int]]): list of flights with source, destination, and cost
            src (int): starting location
            dst (int): desired ending location
            k (int): maximum allowed stops

        Returns:
            int: cheapest price possible using at most k stops
        """
        # Create an adjacency list
        graph = [[] for _ in range(n)]
        for flight in flights:
            depart_location = flight[0]
            arrive_location = flight[1]
            price = flight[2]
            graph[depart_location].append([arrive_location, price])
        
        # Now perform BFS
        record = 2000 * (k+1)
        stops = 0
        bfs_queue = Queue()
        # Each 'record' pushed onto the queue records the location ended up at for this record, and the price
        bfs_queue.push([src, 0])
        while len(bfs_queue) > 0 and stops <= k:
            num_to_dequeue = len(bfs_queue)
            for _ in range(num_to_dequeue):
                next = bfs_queue.pop()
                location = next[0]
                cost = next[1]
                for connection in graph[location]:
                    next_location = connection[0]
                    flight_cost = connection[1]
                    cost_to_next_location = cost + flight_cost
                    if next_location == dst:
                        record = min(record, cost_to_next_location)
                    else:
                        bfs_queue.push([next_location, cost_to_next_location])
            stops += 1
        
        return record if record < 2000 * (k+1) else -1
    
    def solveNQueens(self, n: int) -> List[List[str]]:
        """The n-queens puzzle is the problem of placing n queens on an n x n chessboard so that no two queens can attack each other.
        A queen in a chessboard can attack horizontally, vertically, and diagonally.
        Given an integer n, return all distinct solutions to the n-queens puzzle.
        Each solution contains a unique board layout where the queen pieces are placed. 
        'Q' indicates a queen and '.' indicates an empty space.
        You may return the answer in any order.

        Args:
            n (int): number of queens and board size

        Returns:
            List[List[str]]: all possible arrangements of queens such that they cannot attack each other
        """
        pass
    
    def maxProfit(self, prices: List[int]) -> int:
        """You are given an integer array prices where prices[i] is the price of NeetCoin on the ith day.
        
        You may buy and sell one NeetCoin multiple times with the following restrictions:
        - After you sell your NeetCoin, you cannot buy another one on the next day (i.e., there is a cooldown period of one day).
        - You may only own at most one NeetCoin at a time.
        - You may complete as many transactions as you like.
        
        Return the maximum profit you can achieve.

        Args:
            prices (List[int]): list of stock prices on each day

        Returns:
            int: maximum profit achievable
        """
        n = len(prices)
        # Max profit achievable from here given need to buy
        sol = [-1 for _ in prices]
        # Max profit achievable from here given need to sell
        sell = [-1 for _ in prices]
        
        # Base cases
        sol[n-1] = 0
        sell[n-1] = prices[n-1] # Since prices will be stictly positive
        
        for i in range(n-2, -1, -1):
            price = prices[i]
            
            # If we need to buy from here, try buying, and try not buying
            sol[i] = max(sell[i+1] - price, sol[i+1])
            
            # If we need to sell from here, try selling now, and try selling later
            sell[i] = max(price + (sol[i+2] if i<n-2 else 0), sell[i+1])
        
        return sol[0]