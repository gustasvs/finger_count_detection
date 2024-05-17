class BPlusTree:
    """A simple B+ tree implementation where each node can contain up to 3 keys and 4 children."""

    class Node:
        def __init__(self, keys=None, children=None):
            self.keys = keys or []
            self.children = children or []
            self.is_leaf = not self.children

    def __init__(self):
        self.root = self.Node()

    def insert(self, key):
        node, parent = self._find_node(self.root, None, key)
        self._insert_into_node(node, parent, key)

    def _find_node(self, node, parent, key):
        """Find the node where the key should be inserted."""
        if node.is_leaf:
            return node, parent
        for i, item in enumerate(node.keys):
            if key < item:
                return self._find_node(node.children[i], node, key)
        return self._find_node(node.children[-1], node, key)

    def _insert_into_node(self, node, parent, key):
        """Insert a key into a node and handle splitting if needed."""
        node.keys.append(key)
        node.keys.sort()
        if len(node.keys) > 3:
            # Node is full; split the node
            middle = len(node.keys) // 2
            left_keys = node.keys[:middle]
            right_keys = node.keys[middle + 1:]
            middle_key = node.keys[middle]

            # Create new nodes for the split
            left = self.Node(left_keys)
            right = self.Node(right_keys)

            if node.is_leaf:
                # Maintain linked list property for leaf nodes
                right.children = node.children
                left.children = [right]

            if parent is None:
                # The node was the root; create a new root
                self.root = self.Node(keys=[middle_key], children=[left, right])
            else:
                # Insert the middle key into the parent
                self._insert_into_node(parent, None, middle_key)
                # Update parent's children references
                index = parent.children.index(node)
                parent.children[index:index + 1] = [left, right]

    def load_structure(self, structure):
        """Load a nested array structure into the B+ tree."""
        def create_node_from_structure(struct):
            if isinstance(struct[0], list):
                # This is a non-leaf node
                keys = [item for item in struct if isinstance(item, int)]
                children_structures = [item for item in struct if isinstance(item, list)]
                children = [create_node_from_structure(child) for child in children_structures]
                return self.Node(keys, children)
            else:
                # This is a leaf node
                return self.Node(struct)

        self.root = create_node_from_structure(structure)

    def display(self):
        """Display the B+ tree as a nested array."""
        def recurse(node):
            if node.is_leaf:
                return node.keys
            else:
                res = []
                for child, key in zip(node.children, node.keys + [None]):
                    res.extend(recurse(child))
                    if key:
                        res.append(key)
                return res
        return recurse(self.root)

# Initialize a BPlusTree instance
tree = BPlusTree()

# Given tree structure
tree_structure = [[[2, 8, 24], 32, [32, 34, 38]], 58, [58, 62, 64], 72, [72, 88, 98]]

# Load the structure into the BPlusTree
tree.load_structure(tree_structure)

# Insert the number 17 into the BPlusTree
tree.insert(17)

# Display the final structure of the tree
final_structure = tree.display()
print(final_structure)
