"""Deduplicate candidate clips by visual similarity and time overlap."""
from . import config, utils


class DedupFilter:
    def __init__(self, sim_threshold=None):
        self.sim_threshold = sim_threshold if sim_threshold is not None \
            else config.SIMILARITY_THRESHOLD
        # Each kept item: dict(source, start, end, hash)
        self.kept = []

    def is_duplicate(self, source, start, end, frame_hash):
        # 1) Time-overlap dedup within the same source video.
        #    仅当时间区间完全相同才判重（同一帧），避免同一视频内
        #    不同时间点的不同画面被误判为重复。
        for k in self.kept:
            if (k["source"] == source and
                    start == k["start"] and end == k["end"]):
                return True, "time-overlap"
        # 2) Visual-similarity dedup via perceptual hash.
        #    跨视频和同视频内的相似画面都靠这个判断。
        if frame_hash is not None:
            for k in self.kept:
                if k["hash"] is None:
                    continue
                if utils.hamm(k["hash"], frame_hash) <= self.sim_threshold:
                    return True, "visual-similar"
        return False, None

    def add(self, source, start, end, frame_hash):
        self.kept.append({
            "source": source, "start": start, "end": end, "hash": frame_hash,
        })
