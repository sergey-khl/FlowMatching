import numpy as np
from typing import Optional

class SequenceSamplerAsync:
    """
        allow different horizons on different keys
        seperate out vision collection with low_dim.
    """

    def __init__(
        self,
        low_dim_replay_buffer,
        img_replay_buffer,
        obs_key_horizons: dict,
        action_horizon: int = 1,
        action_down_sample_steps: int = 1,
        action_padding: bool = False,
        low_dim_time_key: str = "robot_timestamp_ns",
        img_time_key: str = "img_timestamp_ns",
        episode_mask: Optional[np.ndarray] = None,
        query_frequency_down_sample_steps: int = 1,
        buffer_time_ms: float = 200.0,
        correction_upsampling: bool = False,
        correction_weight_threshold: float = 1.0,
        correction_lookback_horizon: int = 1,
        correction_lookback_down_sample_steps: int = 1,
        weighted_sampling: int = 1,
        correction_horizon: int = 0,
        only_correction: bool = False,
    ):
        self.low_dim_replay_buffer = low_dim_replay_buffer
        self.img_replay_buffer = img_replay_buffer
        self.obs_key_horizons = obs_key_horizons
        self.action_horizon = action_horizon
        self.action_down_sample_steps = action_down_sample_steps
        self.action_padding = action_padding
        self.low_dim_time_key = low_dim_time_key
        self.img_time_key = img_time_key
        self.buffer_time_ns = buffer_time_ms * 1e6

        low_dim_ends = low_dim_replay_buffer.episode_ends[:]
        img_ends = img_replay_buffer.episode_ends[:]
        n_episodes = len(low_dim_ends)

        low_dim_starts = np.concatenate([[0], low_dim_ends[:-1]])
        img_starts = np.concatenate([[0], img_ends[:-1]])
        self.low_dim_starts = low_dim_starts
        self.low_dim_ends = low_dim_ends
        self.img_starts = img_starts
        self.img_ends = img_ends
 
        low_dim_time = low_dim_replay_buffer[low_dim_time_key]
        img_time = img_replay_buffer[img_time_key]
 
        if episode_mask is None:
            episode_mask = np.ones(n_episodes, dtype=bool)
 
        indices = []  # (episode_idx, img_query_idx)
        valid_ranges = {}

        for ep in range(n_episodes):
            if not episode_mask[ep]:
                continue
 
            ld_start, ld_end = int(low_dim_starts[ep]), int(low_dim_ends[ep])
            im_start, im_end = int(img_starts[ep]), int(img_ends[ep])
            if ld_end <= ld_start or im_end <= im_start:
                continue
 
            # furthest up obs
            start_time = -np.inf
            for key, cfg in obs_key_horizons.items():
                horizon = cfg["horizon"]
                down_sample_steps = cfg.get("down_sample_steps", 1)
                source = cfg.get("source", "low_dim")
                start_offset = horizon * down_sample_steps + 1
                if source == "low_dim":
                    idx = min(ld_start + start_offset, ld_end - 1)
                    candidate = low_dim_time[idx]
                else:  # img
                    idx = min(im_start + start_offset, im_end - 1)
                    candidate = img_time[idx]
                start_time = max(start_time, candidate)
 
            # farthest back obs
            low_dim_end_time = low_dim_time[ld_end - 1]
            img_end_time = img_time[im_end - 1]
            end_time = min(low_dim_end_time, img_end_time)
            if not action_padding:
                end_time = end_time - self.buffer_time_ns
 
            assert end_time > start_time
 
            local_img_time = img_time[im_start:im_end]
            last_img_idx = im_start + int(
                np.searchsorted(local_img_time, end_time, side="right") - 1
            )
            first_img_idx = im_start + int(
                np.searchsorted(local_img_time, start_time, side="left")
            )
            assert last_img_idx > first_img_idx
 
            valid_ranges[ep] = (first_img_idx, last_img_idx)
            if not only_correction:
                ids_to_be_added = np.arange(first_img_idx, last_img_idx)
                for q in ids_to_be_added[::query_frequency_down_sample_steps]:
                    indices.append((ep, int(q)))

        if correction_upsampling:
            indices.extend(
                self._build_correction_indices(
                    valid_ranges=valid_ranges,
                    query_frequency_down_sample_steps=query_frequency_down_sample_steps,
                    threshold=correction_weight_threshold,
                    lookback_horizon=correction_lookback_horizon,
                    lookback_down_sample_steps=correction_lookback_down_sample_steps,
                    weighted_sampling=weighted_sampling,
                    correction_horizon=correction_horizon,
                    only_correction=only_correction,
                )
            )
        self.indices = indices

    def _nearest_low_dim_ids(self, ep, query_times):
        """Global low_dim row indices nearest-at-or-after each of query_times, for episode ep."""
        ld_start = int(self.low_dim_starts[ep])
        ld_end = int(self.low_dim_ends[ep])
        low_dim_time = self.low_dim_replay_buffer[self.low_dim_time_key]
        local_time = low_dim_time[ld_start:ld_end]
        local_idx = np.searchsorted(local_time, query_times)
        local_idx = np.clip(local_idx, 0, len(local_time) - 1)
        return ld_start + local_idx

    def _build_correction_indices(
        self,
        valid_ranges,
        query_frequency_down_sample_steps,
        threshold,
        lookback_horizon,
        lookback_down_sample_steps,
        weighted_sampling,
        correction_horizon,
        only_correction,
    ):
        extra = []
        traj_weight = self.low_dim_replay_buffer["traj_weight"]
        img_time = self.img_replay_buffer[self.img_time_key]
        lookback_span = lookback_horizon * lookback_down_sample_steps

        for ep, (first_img_idx, last_img_idx) in valid_ranges.items():
            img_query_ids = np.arange(first_img_idx, last_img_idx)[
                ::query_frequency_down_sample_steps
            ]
            if len(img_query_ids) == 0:
                continue

            ld_start = int(self.low_dim_starts[ep])
            ld_end = int(self.low_dim_ends[ep])
 
            # align each img query to its nearest low_dim row before touching traj_weight
            query_times = img_time[img_query_ids]
            low_dim_query_ids = self._nearest_low_dim_ids(ep, query_times)
 
            sample_ids = np.clip(low_dim_query_ids - lookback_span, ld_start, ld_end - 1)
            flag_not_in_correction = traj_weight[sample_ids] < threshold
 
            lookback_range = np.arange(0, lookback_span, max(lookback_down_sample_steps, 1))
            lookback_ids = np.clip(
                sample_ids[:, None] + lookback_range, ld_start, ld_end - 1
            )
            flag_before_correction = np.mean(traj_weight[lookback_ids], axis=-1) > threshold
 
            is_correction_sample = flag_not_in_correction & flag_before_correction
            correction_img_ids = img_query_ids[is_correction_sample]
 
            in_correction_img_ids = img_query_ids[traj_weight[low_dim_query_ids] > threshold]
 
            for _ in range(weighted_sampling):
                for cid in correction_img_ids:
                    extra.append((ep, int(cid)))
 
            if only_correction:
                print(f"[episode {ep}] total correction sample num {len(in_correction_img_ids)}")
                for cid in in_correction_img_ids:
                    extra.append((ep, int(cid)))
 
            if correction_horizon > 0 and len(correction_img_ids) > 0:
                addition_horizon = np.arange(
                    query_frequency_down_sample_steps,
                    (correction_horizon + 1) * query_frequency_down_sample_steps,
                    query_frequency_down_sample_steps,
                )
                additional_sample_ids = np.clip(
                    correction_img_ids[:, None] + addition_horizon,
                    img_query_ids[0], img_query_ids[-1],
                )
                additional_sample_ids = np.unique(additional_sample_ids.reshape(-1))
 
                additional_sample_ids = additional_sample_ids[
                    np.isin(additional_sample_ids, in_correction_img_ids)
                ]
                additional_sample_ids = additional_sample_ids[
                    ~np.isin(additional_sample_ids, correction_img_ids)
                ]
 
                for _ in range(weighted_sampling):
                    for aid in additional_sample_ids:
                        extra.append((ep, int(aid)))
 
        return extra

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _slice_backward(arr, center_idx, ep_start, horizon, down_sample_steps):
        """`horizon` frames ending at (and including) center_idx, strided by
        down_sample_steps, going backwards. Pads by repeating the earliest
        in-window frame if the episode doesn't have enough history."""
        offsets = np.arange(-(horizon - 1) * down_sample_steps, 1, down_sample_steps)
        raw_ids = center_idx + offsets
        clipped_ids = np.clip(raw_ids, ep_start, center_idx)
        return arr[clipped_ids]

    @staticmethod
    def _slice_forward(arr, start_idx, ep_end, horizon, down_sample_steps, padding):
        """`horizon` frames starting at start_idx, strided by down_sample_steps,
        going forwards. Pads by repeating the last in-episode frame if
        padding=True; otherwise the caller must have already guaranteed
        there's enough room (see valid_end computation in __init__)."""
        offsets = np.arange(0, horizon * down_sample_steps, down_sample_steps)
        raw_ids = start_idx + offsets
        if padding:
            raw_ids = np.clip(raw_ids, start_idx, ep_end - 1)
        return arr[raw_ids]

    def sample_sequence(self, idx):
        ep, img_query_idx = self.indices[idx]
        low_dim_ep_start = int(self.low_dim_starts[ep])
        low_dim_ep_end = int(self.low_dim_ends[ep])
        img_ep_start = int(self.img_starts[ep])
 
        query_time = self.img_replay_buffer[self.img_time_key][img_query_idx]
        low_dim_query_idx = int(
            self._nearest_low_dim_ids(ep, np.array([query_time]))[0]
        )
 
        obs_result = {}
        for key, cfg in self.obs_key_horizons.items():
            horizon = cfg["horizon"]
            down_sample_steps = cfg.get("down_sample_steps", 1)
            source = cfg.get("source", "low_dim")
 
            if source == "low_dim":
                arr = self.low_dim_replay_buffer[key]
                obs_result[key] = self._slice_backward(
                    arr, low_dim_query_idx, low_dim_ep_start, horizon, down_sample_steps
                )
            else:  # img
                arr = self.img_replay_buffer[key]
                obs_result[key] = self._slice_backward(
                    arr, img_query_idx, img_ep_start, horizon, down_sample_steps
                )
 
        action_arr = self.low_dim_replay_buffer["action"]
        action = self._slice_forward(
            action_arr,
            low_dim_query_idx,
            low_dim_ep_end,
            self.action_horizon,
            self.action_down_sample_steps,
            padding=self.action_padding,
        )
 
        return obs_result, action
