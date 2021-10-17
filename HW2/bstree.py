import numpy as np
import random
import math

class Node:
    def __init__(self, key, value = -1):
        self.left = None
        self.right = None
        self.key = key
        self.value = value

def insert(root, key, value):
    if root is None:
        root = Node(key, value)
        # print(root.key)
    else:
        if key < root.key:
            # print(key)
            root.left = insert(root.left, key, value)
        elif key > root.key:
            root.right = insert(root.right, key, value)
        else:
            pass
    return root

def bstree_construction(db_np):
    root = None
    for i, point in enumerate(db_np):
        # print(i, point)
        root = insert(root, point, i)
    return root

def search_recursive(root, key):
    if root is None or root.key == key:
        return root
    if key < root.key:
        return search_recursive(root.left, key)
    elif key > root.key:
        return search_recursive(root.right, key)

def inorder(root):
    # print('1')
    if root is not None:
        inorder(root.left)
        print(root.key)
        inorder(root.right)

def traverse_bstree(root, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.left is not None:
        traverse_bstree(root.left, depth, max_depth)
    if root.right is not None:
        traverse_bstree(root.right, depth, max_depth)

    depth[0] -= 1

def main():
    # configuration
    db_size = 2048
    dim = 1

    db_np = np.random.rand(db_size, dim)
    # print(db_np)

    root = bstree_construction(db_np)
    # inorder(root)
    depth = [0]
    max_depth = [0]
    traverse_bstree(root, depth, max_depth)
    rootn = search_recursive(root, db_np[0, -1])
    # print(rootn.key)
    print("tree max depth: %d" % max_depth[0])

if __name__ == '__main__':
    main()