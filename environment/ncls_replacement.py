# ncls_replacement.py
# A drop-in replacement for NCLS using intervaltree on Windows.

from intervaltree import IntervalTree

class NCLS:
    def __init__(self, starts, ends, ids):
        self.tree = IntervalTree()
        for s, e, idx in zip(starts, ends, ids):
            # intervaltree uses [begin, end)
            # but NCLS uses [start, end], so we add +1
            self.tree[s:e+1] = idx

    def find_overlap(self, start, end):
        # NCLS uses inclusive, intervaltree is [start, end)
        overlaps = self.tree.overlap(start, end)

        result = []
        for interval in overlaps:
            # NCLS returns (start, end, id)
            result.append((interval.begin, interval.end - 1, interval.data))

        return result
