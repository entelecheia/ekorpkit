#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-07-24 11:30:23
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

"""Deduplicating Training Data Makes Language Models Better."""

import numpy as np
import multiprocessing as mp
from typing import List, Any, Tuple
from multiprocessing import Manager
from ctypes import c_char_p
from numpy.lib.stride_tricks import sliding_window_view
from nltk.util import ngrams


class MinHashDeduper:
    def __init__(
        self, num_perm: int = 128, threshold: float = 0.5, ngram_size: int = 5
    ):

        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.lsh = None

    def fit_transform(self, data: List[str]) -> List[int]:
        from datasketch import MinHash, MinHashLSH

        """Group similar documents with minhash.

        Parameters
        ----------
        data : List[str]
            List of document strings.

        Returns
        -------
        List[int]
            List of group indices.
        
        Examples
        --------
        >>> deduper = MinHashDeduper(ngram_size=5, threshold=0.3)
        >>> groups = deduper.fit_transform(["This is a sentence.", "This is another sentence.", "This is a question.", "hello world"])
        >>> groups
        [0, 0, 2, 3]
        """
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        signatures = []
        for i, doc in enumerate(data):
            m = MinHash(num_perm=self.num_perm)
            for ngram in ngrams(doc, self.ngram_size):
                m.update("".join(ngram).encode("utf-8"))
            signatures.append(m)
            self.lsh.insert(f"m{i}", m)

        neighbors = []
        for i, doc in enumerate(data):
            result = self.lsh.query(signatures[i])
            neighbors.append([int(x[1:]) for x in result])

        return get_group_indices(neighbors)


class SuffixArray:
    def __init__(self, k: int = 50):
        self.array = []
        self.k = k

    def fit_transform(self, data: List[str]) -> Tuple[List[str], np.ndarray]:
        """Find duplicate substrings in the data.

        Parameters
        ----------
        data : List[str]
            List of documents.

        Returns
        -------
        Tuple[List[str], np.ndarray]
            List of duplicate substrings and a matrix where each row is a document and each column is a substring.

        Examples
        --------
        >>> array = SuffixArray(k = 9)
        >>> duplicates, groups = array.fit_transform(["This is a sentence.", "This is another sentences.", "This is a question.", "hello world"] * 10)
        >>> assert len(duplicates) == groups.shape[1], "Invalid number of columns"
        >>> assert groups.shape[0] == 40, "Invalid number of rows"
        """
        S = "".join(data)
        suffixes = []
        for i in range(len(S)):
            suffixes.append(S[i:])

        self.array = np.argsort(suffixes)

        # Find duplicated substrings
        manager = Manager()
        shared = manager.Value(c_char_p, S)

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(
                similar,
                [(x, y, shared, self.k) for x, y in sliding_window_view(self.array, 2)],
            )

        duplicates = []
        for idx, dup in zip(self.array, results):
            if dup:
                duplicates.append(S[idx : idx + self.k])

        # Find duplicated documents
        try:
            from multiprocessing import shared_memory

            shared = shared_memory.ShareableList(duplicates)
        except ImportError as e:
            print(
                f"The following error was: \n{e}\n\n"
                + "This was likely raised since you are not running python 3.8 or higher."
                + " Continuing without a shared memory file which is likely be inefficient."
            )
            shared = duplicates
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(group, [(d, shared) for d in data])

        shared.shm.close()
        shared.shm.unlink()
        del shared

        groups = np.zeros((len(data), len(duplicates)), dtype=bool)
        for i, x in enumerate(results):
            for y in x:
                groups[i, y] = 1

        return duplicates, groups


def similar(x: int, y: int, S: Any, k: int) -> bool:
    """Whether S[x:x+k] is the same as S[y:y+k].

    Parameters
    ----------
    x : int
        [description]
    y : int
        [description]
    S : Any
        [description]
    k : int
        [description]

    Returns
    -------
    bool
        [description]
    """
    if x == y:
        return True

    return (
        x + k <= len(S.value)
        and y + k <= len(S.value)
        and S.value[x : x + k] == S.value[y : y + k]
    )


def group(x: str, patterns: str) -> List[int]:
    """Find patterns that are present in string x.

    Parameters
    ----------
    x : str
        A document string
    patterns : str
        Patterns to search for

    Returns
    -------
    List[int]
        List of indices of which patterns are present in string x
    """
    result = []
    for idx, pattern in enumerate(patterns):
        if pattern in x:
            result.append(idx)
    return result


def get_group_indices(neighbors: List[List[int]]) -> List[int]:
    """Based on the nearest neighbors, find the group/cluster index for each element.

    Parameters
    ----------
    neighbors : List[List[int]]
        List of nearest neighbor indices

    Returns
    -------
    List[int]
        List of group indices
    """
    finder = UF(len(neighbors))
    for i, n in enumerate(neighbors):
        for j in n:
            finder.union(i, j)

    return [finder.find(i) for i in range(len(neighbors))]


"""This module implements an union find or disjoint set data structure.

Source: https://python-algorithms.readthedocs.io/en/stable/_modules/python_algorithms/basic/union_find.html

An union find data structure can keep track of a set of elements into a number
of disjoint (nonoverlapping) subsets. That is why it is also known as the
disjoint set data structure. Mainly two useful operations on such a data
structure can be performed. A *find* operation determines which subset a
particular element is in. This can be used for determining if two
elements are in the same subset. An *union* Join two subsets into a
single subset.

The complexity of these two operations depend on the particular implementation.
It is possible to achieve constant time (O(1)) for any one of those operations
while the operation is penalized. A balance between the complexities of these
two operations is desirable and achievable following two enhancements:

1.  Using union by rank -- always attach the smaller tree to the root of the
    larger tree.
2.  Using path compression -- flattening the structure of the tree whenever
    find is used on it.

complexity:
    * find -- :math:`O(\\alpha(N))` where :math:`\\alpha(n)` is
      `inverse ackerman function
      <http://en.wikipedia.org/wiki/Ackermann_function#Inverse>`_.
    * union -- :math:`O(\\alpha(N))` where :math:`\\alpha(n)` is
      `inverse ackerman function
      <http://en.wikipedia.org/wiki/Ackermann_function#Inverse>`_.

"""


class UF:  # pragma: no cover
    """An implementation of union find data structure.
    It uses weighted quick union by rank with path compression.
    """

    def __init__(self, N):
        """Initialize an empty union find object with N items.

        Args:
            N: Number of items in the union find object.
        """

        self._id = list(range(N))
        self._count = N
        self._rank = [0] * N

    def find(self, p):
        """Find the set identifier for the item p."""

        id = self._id
        while p != id[p]:
            p = id[p] = id[id[p]]  # Path compression using halving.
        return p

    def count(self):
        """Return the number of items."""

        return self._count

    def connected(self, p, q):
        """Check if the items p and q are on the same set or not."""

        return self.find(p) == self.find(q)

    def union(self, p, q):
        """Combine sets containing p and q into a single set."""

        id = self._id
        rank = self._rank

        i = self.find(p)
        j = self.find(q)
        if i == j:
            return

        self._count -= 1
        if rank[i] < rank[j]:
            id[i] = j
        elif rank[i] > rank[j]:
            id[j] = i
        else:
            id[j] = i
            rank[i] += 1

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._id])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"
