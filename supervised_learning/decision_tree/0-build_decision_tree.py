#!/usr/bin/env python3
"""0-build_decision_tree"""


import numpy as np


class Node():
    """class Node"""
    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0
            ):
        """Initializing attributes"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Method for finding maximum node depths"""
        if self.is_leaf:
            return self.depth
        else:
            if self.left_child and self.right_child:
                left_depth = self.left_child.max_depth_below()
                right_depth = self.right_child.max_depth_below()
                return max(left_depth, right_depth)


class Leaf(Node):
    """class leaf"""
    def __init__(self, value, depth=None):
        """Initializing attributes"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Maximum node depth"""
        return self.depth


class Decision_Tree():
    """class decision tree"""
    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None
            ):
        """Initializing attributes"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Depth"""
        return self.root.max_depth_below()
