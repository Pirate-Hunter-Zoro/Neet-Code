class ListNode:
    
    def __init__(self, values:list[any]=[]):
        self.val = None
        self.next = None
        if len(values) > 1:
            self.val = values[0]
            self.next = ListNode(values=values[1:])
        elif len(values) == 1:
            self.val = values[0]
            

class Queue:
    pass