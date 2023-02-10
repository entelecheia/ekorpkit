from copy import deepcopy


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

    def get_value_pos(self, word, pos):
        if self.direction == "backward":
            # reverse the word
            word = word[::-1]
        char = word[pos]
        return self.get_value(char)

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
        return node

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
        node = deepcopy(node)
        if not node:
            return []
        return self._get_leafs(node)

    def _get_leafs(self, node):
        if self.end_symbol in node and len(node) == 1:
            return [node[self.end_symbol]]
        elif self.end_symbol in node and len(node) > 1:
            _ = node.pop(self.end_symbol)
        leafs = []
        for child in node:
            leafs += self._get_leafs(node[child])
        return leafs
