import math
from scipy.special import digamma


def entropy(trie, word):
    leafs = trie.get_leafs(word)
    val = trie.get_value(word)
    logsum = digamma(sum(leafs) + val)
    entropy = 0
    for freq in leafs:
        logprob = digamma(freq) - logsum
        entropy += math.exp(logprob) * logprob
    return -1 * entropy


class Trie:
    def __init__(self, end_symbol="<END>", direction="forward"):
        self.root = {}
        self.end_symbol = end_symbol
        self.direction = direction

    def add(self, word, value):
        if self.direction == "backward":
            # reverse the word
            word = word[::-1]
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node[self.end_symbol] = value

    def get_value(self, word):
        if self.direction == "backward":
            # reverse the word
            word = word[::-1]
        node = self.root
        for ch in word:
            if ch not in node:
                return 0
            node = node[ch]
        if self.end_symbol not in node:
            return 0
        return node[self.end_symbol]

    def set_value(self, word, value):
        if self.direction == "backward":
            # reverse the word
            word = word[::-1]
        node = self.root
        for ch in word:
            if ch not in node:
                raise ValueError("word not in trie")
            node = node[ch]
        if self.end_symbol not in node:
            raise ValueError("word not in trie")
        node[self.end_symbol] = value

    def get_children(self, word):
        if self.direction == "backward":
            # reverse the word
            word = word[::-1]
        node = self.root
        for ch in word:
            if ch not in node:
                return []
            node = node[ch]
        children = node.copy()
        return children

    def get_values_of_children(self, word):
        children = self.get_children(word)
        values = []
        for child in children:
            if child == self.end_symbol:
                continue
            else:
                if self.end_symbol in children[child]:
                    values.append(children[child][self.end_symbol])
        return values

    def num_children(self, word):
        return len(self.get_children(word))

    def get_leafs(self, word):
        node = self.get_children(word)
        if not node:
            return []
        if self.end_symbol in node:
            _ = node.pop(self.end_symbol)
        return self._get_leafs(node)

    def _get_leafs(self, node):
        if self.end_symbol in node:
            return [node[self.end_symbol]]
        leafs = []
        for child in node:
            leafs += self._get_leafs(node[child])
        return leafs
