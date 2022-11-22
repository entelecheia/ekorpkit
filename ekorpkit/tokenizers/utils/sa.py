# https://tryalgo.org/en/strings/2021/02/15/count-distinct-substrings/
# https://louisabraham.github.io/notebooks/suffix_arrays.html
from itertools import zip_longest, islice
from random import randint
from collections import defaultdict
from collections import Counter


def contant_string(length):
    return "a" * int(length)


def random_string(length):
    return "".join(chr(randint(0, 255)) for _ in range(int(length)))


def to_int_keys(l):
    """
    l: iterable of keys
    returns: a list with integer keys
    """
    seen = set()
    ls = []
    for e in l:
        if e not in seen:
            ls.append(e)
            seen.add(e)
    ls.sort()
    index = {v: i for i, v in enumerate(ls)}
    return [index[v] for v in l]


def suffix_matrix(s):
    """
    suffix matrix of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys(s)
    ans = [line]
    while max(line) < n - 1:
        line = to_int_keys(
            [
                a * (n + 1) + b + 1
                for (a, b) in zip_longest(line, islice(line, k, None), fillvalue=-1)
            ]
        )
        ans.append(line)
        k <<= 1
    return ans


def suffix_array(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys(s)
    while max(line) < n - 1:
        line = to_int_keys(
            [
                a * (n + 1) + b + 1
                for (a, b) in zip_longest(line, islice(line, k, None), fillvalue=-1)
            ]
        )
        k <<= 1
    return line


def longest_common_prefix(sm, i, j):
    """
    longest common prefix
    O(log(n))

    sm: suffix matrix
    """
    n = len(sm[-1])
    if i == j:
        return n - i
    k = 1 << (len(sm) - 2)
    ans = 0
    for line in sm[-2::-1]:
        if i >= n or j >= n:
            break
        if line[i] == line[j]:
            ans ^= k
            i += k
            j += k
        k >>= 1
    return ans


class SuffixArray:
    """by Karp, Miller, Rosenberg 1972

    s is the string to analyze.

    P[k][i] is the pseudo rank of s[i:i+K] for K = 1<<k
    among all strings of length K. Pseudo, because the pseudo rank numbers are
    in order but not necessarily consecutive.

    Initialization of the data structure has complexity O(n log^2 n).
    """

    def __init__(self, s):
        self.n = len(s)
        if self.n == 1:  # special case: single char strings
            self.P = [[0]]
            self.suf_sorted = [0]
            return
        self.P = [list(map(ord, s))]
        k = 1
        length = 1  # length is 2 ** (k - 1)
        while length < self.n:
            L = []  # prepare L
            for i in range(self.n - length):
                L.append((self.P[k - 1][i], self.P[k - 1][i + length], i))
            for i in range(self.n - length, self.n):  # pad with -1
                L.append((self.P[k - 1][i], -1, i))
            L.sort()  # bucket sort would be quicker
            self.P.append([0] * self.n)  # produce k-th row in P
            for i in range(self.n):
                if i > 0 and L[i - 1][:2] == L[i][:2]:  # same as previous
                    self.P[k][L[i][2]] = self.P[k][L[i - 1][2]]
                else:
                    self.P[k][L[i][2]] = i
            k += 1
            length <<= 1  # or *=2 as you prefer
        self.suf_sorted = [0] * self.n  # generate the inverse:
        for i, si in enumerate(self.P[-1]):  # lexic. sorted suffixes
            self.suf_sorted[si] = i

    def longest_common_prefix(self, i, j):
        """returns the length of
        the longest common prefix of s[i:] and s[j:].
        complexity: O(log n), for n = len(s).
        """
        if i == j:
            return self.n - i  # length of suffix
        answer = 0
        length = 1 << (len(self.P) - 1)  # length is 2 ** k
        for k in range(len(self.P) - 1, -1, -1):
            length = 1 << k
            if self.P[k][i] == self.P[k][j]:  # aha, s[i:i+length] == s[j:j+length]
                answer += length
                i += length
                j += length
                if i == self.n or j == self.n:  # not needed if s is appended by $
                    break
            length >>= 1
        return answer

    def number_substrings(self):
        answer = self.n - self.suf_sorted[0]
        for i in range(1, self.n):
            r = self.longest_common_prefix(self.suf_sorted[i - 1], self.suf_sorted[i])
            answer += self.n - self.suf_sorted[i] - r
        return answer


def minimal_lexicographical_rotation(s):
    """returns i such that s[i:]+s[:i] is minimal,
    for n = len(s).
    Could an be solved in linear time,
    but this is an easy O(n log^2 n) solution.
    Uses the observation, that solution i also minimizes (s+s)[i:]
    """
    A = SuffixArray(s + s)
    # find index 0 <= i < len(s) with smallest rank
    best = 0
    for i in range(1, len(s)):
        if A.P[-1][i] < A.P[-1][best]:
            best = i
    return best


def sort_bucket(s, bucket, order):
    d = defaultdict(list)
    for i in bucket:
        key = s[i : i + order]
        d[key].append(i)
    result = []
    for k, v in sorted(d.items()):
        if len(v) > 1:
            result += sort_bucket(s, v, order * 2)
        else:
            result.append(v[0])
    return result


def suffix_array_ManberMyers(s):
    return sort_bucket(s, (i for i in range(len(s))), 1)


def lcp_array(s, sa):
    # http://codeforces.com/blog/entry/12796
    n = len(s)
    k = 0
    lcp = [0] * n
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i

    for i in range(n):
        if rank[i] == n - 1:
            k = 0
            continue
        j = sa[rank[i] + 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k:
            k -= 1

    return lcp


# https://github.com/glickmac/Burrows_Wheeler_in_Python


def suffix_array2(string):
    return list(sorted(range(len(string)), key=lambda i: string[i:]))


def bwt_from_suffix(string, s_array=None):
    if s_array is None:
        s_array = suffix_array2(string)
    return "".join(string[idx - 1] for idx in s_array)


def lf_mapping(bwt, letters=None):
    if letters is None:
        letters = set(bwt)

    result = {letter: [0] for letter in letters}
    result[bwt[0]] = [1]
    for letter in bwt[1:]:
        for i, j in result.items():
            j.append(j[-1] + (i == letter))
    return result


def count_occurences(string, letters=None):
    count = 0
    result = {}

    if letters is None:
        letters = set(string)

    c = Counter(string)

    for letter in sorted(letters):
        result[letter] = count
        count += c[letter]
    return result


def update(begin, end, letter, lf_map, counts, string_length):
    beginning = counts[letter] + lf_map[letter][begin - 1] + 1
    ending = counts[letter] + lf_map[letter][end]
    return (beginning, ending)


def generate_all(input_string, s_array=None, eos="$"):
    letters = set(input_string)
    try:
        assert eos not in letters

        counts = count_occurences(input_string, letters)

        input_string = "".join([input_string, eos])
        if s_array is None:
            s_array = suffix_array(input_string)
        bwt = bwt_from_suffix(input_string, s_array)
        lf_map = lf_mapping(bwt, letters | set([eos]))

        for i, j in lf_map.items():
            j.extend([j[-1], 0])  # for when pointers go off the edges

        return letters, bwt, lf_map, counts, s_array

    except AssertionError:
        print("End of string character found in text, deleted EOS from input string")
        input_string = input_string.replace(eos, "")
        letters = set(input_string)
        counts = count_occurences(input_string, letters)

        input_string = "".join([input_string, eos])
        if s_array is None:
            s_array = suffix_array(input_string)
        bwt = bwt_from_suffix(input_string, s_array)
        lf_map = lf_mapping(bwt, letters | set([eos]))

        for i, j in lf_map.items():
            j.extend([j[-1], 0])  # for when pointers go off the edges

        return letters, bwt, lf_map, counts, s_array


def find(search_string, input_string, mismatches=0, bwt_data=None, s_array=None):

    results = []

    if len(search_string) == 0:
        return "Empty Query String"
    if bwt_data is None:
        bwt_data = generate_all(input_string, s_array=s_array)

    letters, bwt, lf_map, count, s_array = bwt_data

    if len(letters) == 0:
        return "Empty Search String"

    if not set(search_string) <= letters:
        return []

    length = len(bwt)

    class Fuzzy(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fuz = [
        Fuzzy(
            search_string=search_string,
            begin=0,
            end=len(bwt) - 1,
            mismatches=mismatches,
        )
    ]

    while len(fuz) > 0:
        p = fuz.pop()
        search_string = p.search_string[:-1]
        last = p.search_string[-1]
        all_letters = [last] if p.mismatches == 0 else letters
        for letter in all_letters:
            begin, end = update(p.begin, p.end, letter, lf_map, count, length)
            if begin <= end:
                if len(search_string) == 0:
                    results.extend(s_array[begin : end + 1])
                else:
                    miss = p.mismatches
                    if letter != last:
                        miss = max(0, p.mismatches - 1)
                    fuz.append(
                        Fuzzy(
                            search_string=search_string,
                            begin=begin,
                            end=end,
                            mismatches=miss,
                        )
                    )
    return sorted(set(results))
