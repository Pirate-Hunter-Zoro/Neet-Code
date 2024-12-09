class ListNode:
    
    def __init__(self, values:list[any]=[]):
        self.val = None
        self.next = None
        if len(values) > 1:
            self.val = values[0]
            self.next = ListNode(values=values[1:])
        elif len(values) == 1:
            self.val = values[0]
            

class Stack:
    
    def __init__(self):
        self.head = None
        self.length = 0
        
    def __len__(self):
        return self.length
    
    def push(self, v: any):
        self.length += 1
        if self.head == None:
            self.head = ListNode(values=[v])
        else:
            prev_head = self.head
            self.head = ListNode(values=[v])
            self.head.next = prev_head
    
    def pop(self) -> any:
        assert self.length > 0
        self.length -= 1
        v = self.head.val
        self.head = self.head.next
        return v
    
    def peek(self) -> any:
        assert self.length > 0
        return self.head.val