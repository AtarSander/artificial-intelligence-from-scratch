class TreeNode:
    def __init__(self, feature, left=None, right=None, info_gain=None, value=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
