from implementation.heap import Heap


class MedianFinder:
    """The median is the middle value in a sorted list of integers. For lists of even length, there is no middle value, so the median is the mean of the two middle values.
    For example:
    For arr = [1,2,3], the median is 2.
    For arr = [1,2], the median is (1 + 2) / 2 = 1.5
    
    Implement the MedianFinder class:
    MedianFinder() initializes the MedianFinder object.
    void addNum(int num) adds the integer num from the data stream to the data structure.
    double findMedian() returns the median of all elements so far.
    """

    def __init__(self):
        # We're going to need two heaps
        self.__min_heap = Heap()
        self.__max_heap = Heap(comparator = lambda x,y: x > y)

    def addNum(self, num: int) -> None:
        # Determine which heap to add this to
        if len(self.__min_heap) == len(self.__max_heap):
            # Both heaps of equal size
            if self.__min_heap.empty():
                # Both heaps empty
                self.__max_heap.push(num)
            else:
                highest_on_max = self.__max_heap.top()
                if num <= highest_on_max:
                    self.__max_heap.push(num)
                else: # By construction this MUST mean num >= lowest element on min heap
                    self.__min_heap.push(num)
        elif len(self.__min_heap) < len(self.__max_heap):
            # We NEED to give an item to minimum heap
            highest_on_max = self.__max_heap.top()
            if num >= highest_on_max:
                self.__min_heap.push(num)
            else:
                # must take from max heap
                self.__min_heap.push(self.__max_heap.pop())
                self.__max_heap.push(num)
        else:
            # We NEED to give an item to the maximum heap
            lowest_on_min = self.__min_heap.top()
            if num <= lowest_on_min:
                self.__max_heap.push(num)
            else:
                # must take from min heap
                self.__max_heap.push(self.__min_heap.pop())
                self.__min_heap.push(num)

    def findMedian(self) -> float:
        # If the two heaps have an equal number of elements, take average of each top
        if len(self.__max_heap) == len(self.__min_heap):
            return (self.__max_heap.top() + self.__min_heap.top()) / 2
        else: # Take the top of whichever heap has more elements
            if len(self.__max_heap) > len(self.__min_heap):
                return self.__max_heap.top()
            else:
                return self.__min_heap.top()