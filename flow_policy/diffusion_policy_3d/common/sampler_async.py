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
        obs_key_horizons: dict,
        action_horizon: int = 1,
        action_down_sample_steps: int = 1,
        action_padding: bool = False,
        low_dim_time_key: str = "robot_timestamp_ns",
        extra_sources: Optional[dict] = None,
        episode_mask: Optional[np.ndarray] = None,
        query_frequency_down_sample_steps: int = 1,
        correction_upsampling: bool = False,
        correction_weight_threshold: float = 1.0,
        correction_lookback_horizon: int = 1,
        correction_lookback_down_sample_steps: int = 1,
        weighted_sampling: int = 1,
        correction_horizon: int = 0,
        only_correction: bool = False,
    ):
        assert obs_key_horizons is not None and len(obs_key_horizons) > 0

        self.low_dim_replay_buffer = low_dim_replay_buffer
        self.obs_key_horizons = obs_key_horizons
        self.action_horizon = action_horizon
        self.action_down_sample_steps = action_down_sample_steps
        self.action_padding = action_padding
        self.low_dim_time_key = low_dim_time_key
        self.extra_sources = extra_sources or {}


        low_dim_ends = low_dim_replay_buffer.episode_ends[:]
        n_episodes = len(low_dim_ends)

        used_sources = {
            cfg.get("source", "low_dim") for cfg in obs_key_horizons.values()
        }
        for source_name in used_sources:
            if source_name == "low_dim":
                continue

        self.source_starts = {}
        self.source_ends = {}
        for source_name, cfg in self.extra_sources.items():
            source_ends = cfg["replay_buffer"].episode_ends[:]
            assert len(source_ends) == n_episodes, (
                f"low_dim and '{source_name}' have {n_episodes} != {len(source_ends)}"
            )
            self.source_starts[source_name] = np.concatenate([[0], source_ends[:-1]])
            self.source_ends[source_name] = source_ends

        if episode_mask is None:
            episode_mask = np.ones(n_episodes, dtype=bool)

        low_dim_starts = np.concatenate([[0], low_dim_ends[:-1]])
        self.low_dim_starts = low_dim_starts
        self.low_dim_ends = low_dim_ends

        max_action_span = (action_horizon - 1) * action_down_sample_steps

        indices = []  # list of (episode_idx, low_dim_query_idx)
        valid_ranges = {}
        for ep in range(n_episodes):
            if not episode_mask[ep]:
                continue
            ep_start, ep_end = int(low_dim_starts[ep]), int(low_dim_ends[ep])
            if action_padding:
                valid_start, valid_end = ep_start, ep_end
            else:
                valid_start, valid_end = ep_start, ep_end - max_action_span
            if valid_end <= valid_start:
                continue
            valid_ranges[ep] = (valid_start, valid_end)
            if not only_correction:
                ids_to_be_added = np.arange(valid_start, valid_end)
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
        lookback_span = lookback_horizon * lookback_down_sample_steps
 
        for ep, (valid_start, valid_end) in valid_ranges.items():
            ids_to_be_added = np.arange(valid_start, valid_end)
            query_ids = ids_to_be_added[::query_frequency_down_sample_steps]
            if len(query_ids) == 0:
                continue
 
            # position lookback_span frames before each query point
            sample_ids = np.clip(query_ids - lookback_span, valid_start, valid_end - 1)
            flag_not_in_correction = traj_weight[sample_ids] < threshold
 
            lookback_range = np.arange(0, lookback_span, max(lookback_down_sample_steps, 1))
            lookback_ids = np.clip(
                sample_ids[:, None] + lookback_range, valid_start, valid_end - 1
            )
            flag_before_correction = np.mean(traj_weight[lookback_ids], axis=-1) > threshold
 
            is_correction_sample = flag_not_in_correction & flag_before_correction
            correction_ids = query_ids[is_correction_sample]
 
            in_correction_ids = query_ids[traj_weight[query_ids] > threshold]
 
            # repeat some correction samples
            for _ in range(weighted_sampling):
                for cid in correction_ids:
                    extra.append((ep, int(cid)))
 
            if only_correction:
                print(f"[episode {ep}] total correction sample num {len(in_correction_ids)}")
                for cid in in_correction_ids:
                    extra.append((ep, int(cid)))
 
            # PyriteML: if correction_horizon > 0: ... (sample more before/after correction)
            if correction_horizon > 0 and len(correction_ids) > 0:
                addition_horizon = np.arange(
                    query_frequency_down_sample_steps,
                    (correction_horizon + 1) * query_frequency_down_sample_steps,
                    query_frequency_down_sample_steps,
                )
                additional_sample_ids = np.clip(
                    correction_ids[:, None] + addition_horizon,
                    ids_to_be_added[0], ids_to_be_added[-1],
                )
                additional_sample_ids = np.unique(additional_sample_ids.reshape(-1))
 
                additional_sample_ids = additional_sample_ids[
                    np.isin(additional_sample_ids, in_correction_ids)
                ]
                # filter out ones already in correction_ids
                additional_sample_ids = additional_sample_ids[
                    ~np.isin(additional_sample_ids, correction_ids)
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

    def _nearest_source_idx(self, source_name, ep, query_time):
        """Index into `source_name`'s own frame grid (for episode ep) whose
        timestamp is nearest-at-or-after query_time."""
        src_start = int(self.source_starts[source_name][ep])
        src_end = int(self.source_ends[source_name][ep])
        time_key = self.extra_sources[source_name]["time_key"]
        source_rb = self.extra_sources[source_name]["replay_buffer"]
        src_times = source_rb[time_key][src_start:src_end]
        local_idx = np.searchsorted(np.squeeze(src_times), query_time)
        return int(np.clip(src_start + local_idx, src_start, src_end - 1)), src_start
 
    def sample_sequence(self, idx):
        ep, query_idx = self.indices[idx]
        low_dim_ep_start = int(self.low_dim_starts[ep])
        low_dim_ep_end = int(self.low_dim_ends[ep])
        query_time = self.low_dim_replay_buffer[self.low_dim_time_key][query_idx]
 
        obs_result = {}
        for key, cfg in self.obs_key_horizons.items():
            horizon = cfg["horizon"]
            down_sample_steps = cfg.get("down_sample_steps", 1)
            source = cfg.get("source", "low_dim")
 
            if source == "low_dim":
                arr = self.low_dim_replay_buffer[key]
                obs_result[key] = self._slice_backward(
                    arr, query_idx, low_dim_ep_start, horizon, down_sample_steps
                )
            else:
                src_query_idx, src_ep_start = self._nearest_source_idx(
                    source, ep, query_time
                )
                arr = self.extra_sources[source]["replay_buffer"][key]
                obs_result[key] = self._slice_backward(
                    arr, src_query_idx, src_ep_start, horizon, down_sample_steps
                )
 
        action_arr = self.low_dim_replay_buffer["action"]
        action = self._slice_forward(
            action_arr,
            query_idx,
            low_dim_ep_end,
            self.action_horizon,
            self.action_down_sample_steps,
            padding=self.action_padding,
        )
 
        return obs_result, action
