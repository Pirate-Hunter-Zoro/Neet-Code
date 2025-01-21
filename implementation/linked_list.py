class ListNode:
    
    def __init__(self, values:list[any]=[]):
        self.val = None
        self.next = None
        if len(values) > 1:
            self.val = values[0]
            self.next = ListNode(values=values[1:])
        elif len(values) == 1:
            self.val = values[0]
            
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            if self.val != other.val:
                return False
            elif self.next == None and other.next != None:
                return False
            elif self.next != None and other.next == None:
                return False
            else:
                return self.next == other.next
            

class Stack:
    
    def __init__(self):
        self.__head = None
        self.__length = 0
        
    def __len__(self):
        return self.__length
    
    def push(self, v: any):
        self.__length += 1
        if self.__head == None:
            self.__head = ListNode(values=[v])
        else:
            prev_head = self.__head
            self.__head = ListNode(values=[v])
            self.__head.next = prev_head
    
    def pop(self) -> any:
        assert self.__length > 0
        self.__length -= 1
        v = self.__head.val
        self.__head = self.__head.next
        return v
    
    def peek(self) -> any:
        assert self.__length > 0
        return self.__head.val
    
    
class Queue:
    
    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__length = 0
        
    def __len__(self):
        return self.__length
    
    def push(self, v: any):
        self.__length += 1
        if self.__head == None:
            self.__head = ListNode(values=[v])
            self.__tail = self.__head
        else:
            self.__tail.next = ListNode(values=[v])
            self.__tail = self.__tail.next
    
    def pop(self) -> any:
        assert self.__length > 0
        self.__length -= 1
        v = self.__head.val
        self.__head = self.__head.next
        return v
    
    def peek(self) -> any:
        assert self.__length > 0
        return self.__head.val