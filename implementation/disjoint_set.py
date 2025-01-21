class UnionFindNode:
    """The actual node that will assist with union find
    """
    def __init__(self, v:int):
        self.__value = v
        self.__parent = self
        
    def __compress(self):
        """Helper method to compress this node
        """
        if self.__parent != self:
            self.__parent.__compress()
            self.__parent = self.__parent.__parent
    
    def join(self, other) -> bool:
        """Join with the other node

        Args:
            other: other node to join this node with

        Returns:
            bool: whether the two nodes were in separate sets to begin with or not
        """
        self.__compress()
        other.__compress()
        if self.__parent != other.__parent:
            self.__parent.__parent = other.__parent
            self.__compress()
            return True
        else:
            return False
    
    def get_root_value(self) -> int:
        """Getter for this node's value

        Returns:
            int: node's value
        """
        self.__compress()
        return self.__parent.__value
    
    def same_set(self, other) -> bool:
        """Helper method to determine if this node an another node are in the same set

        Args:
            other: other node

        Returns:
            bool: whether the two nodes are in the same set
        """
        self.__compress()
        other.__compress()
        return self.__parent == other.__parent

class UnionFindNodeFactory:
    """Helper class to keep from producing redundant nodes
    """
    def __init__(self):
        self.__map = {}
        
    def make_node(self, id: int):
        """Helper method to make a node if it does not already exist

        Args:
            id (int): id of the node to create
        """
        if id not in self.__map.keys():
            self.__map[id] = UnionFindNode(v=id)
        
    def get_node(self, id: int) -> UnionFindNode:
        """Helper method to get the node with the given id

        Args:
            id (int): id of the node to retrieve

        Returns:
            UnionFindNode: node with the given id
        """
        if id in self.__map.keys():
            return self.__map[id]
        else:
            return None

class UnionFind:
    """Design a Disjoint Set (aka Union-Find) class.
    """
    def __init__(self, n: int):
        """UnionFind(int n) will initialize a disjoint set of size n.

        Args:
            n (int): size of disjoint set
        """
        self.__node_factory = UnionFindNodeFactory()
        for i in range(n):
            self.__node_factory.make_node(id=i)
        self.__num_components = n

    def find(self, x: int) -> int:
        """int find(int x) will return the root of the component that x belongs to.

        Args:
            x (int): value whose root component we want

        Returns:
            int: root component of said value
        """
        return self.__node_factory.get_node(id=x).get_root_value()

    def isSameComponent(self, x: int, y: int) -> bool:
        """bool isSameComponent(int x, int y) will return whether x and y belong to the same component.

        Args:
            x (int): first component
            y (int): second component

        Returns:
            bool: whether the two components are in the same set
        """
        return self.__node_factory.get_node(id=x).same_set(self.__node_factory.get_node(id=y))
        
    def union(self, x: int, y: int) -> bool:
        """bool union(int x, int y) will union the components that x and y belong to. If they are already in the same component, return false, otherwise return true.

        Args:
            x (int): first component
            y (int): second component

        Returns:
            bool: whether the two were in the same set or not
        """
        if self.__node_factory.get_node(id=x).join(self.__node_factory.get_node(id=y)):
            self.__num_components -= 1
            return True
        else:
            return False

    def getNumComponents(self) -> int:
        """int getNumComponents() will return the number of components in the disjoint set.

        Returns:
            int: number of disjoint sets in the union
        """
        return self.__num_components