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
        img_replay_buffer=None,
        action_horizon: int = 1,
        action_down_sample_steps: int = 1,
        action_padding: bool = False,
        low_dim_time_key: str = "robot_timestamp_ns",
        extra_sources: Optional[dict] = None,
        img_time_key: str = "image_timestamp_ns",
        episode_mask: Optional[np.ndarray] = None,
    ):
        assert obs_key_horizons is not None and len(obs_key_horizons) > 0

        self.low_dim_replay_buffer = low_dim_replay_buffer
        self.img_replay_buffer = img_replay_buffer
        self.obs_key_horizons = obs_key_horizons
        self.action_horizon = action_horizon
        self.action_down_sample_steps = action_down_sample_steps
        self.action_padding = action_padding
        self.low_dim_time_key = low_dim_time_key
        self.img_time_key = img_time_key

        low_dim_ends = low_dim_replay_buffer.episode_ends[:]
        n_episodes = len(low_dim_ends)

        needs_img = any(
            cfg.get("source", "low_dim") == "img" for cfg in obs_key_horizons.values()
        )
        if needs_img:
            img_ends = img_replay_buffer.episode_ends[:]
            assert len(img_ends) == n_episodes, (
                f"low_dim and img replay buffers must have the same number of ep"
            )
            self.img_starts = np.concatenate([[0], img_ends[:-1]])
            self.img_ends = img_ends

        if episode_mask is None:
            episode_mask = np.ones(n_episodes, dtype=bool)

        low_dim_starts = np.concatenate([[0], low_dim_ends[:-1]])
        self.low_dim_starts = low_dim_starts
        self.low_dim_ends = low_dim_ends

        max_action_span = (action_horizon - 1) * action_down_sample_steps

        indices = []  # list of (episode_idx, low_dim_query_idx)
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
            for q in range(valid_start, valid_end):
                indices.append((ep, q))

        self.indices = indices

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
        ep, query_idx = self.indices[idx]
        low_dim_ep_start = int(self.low_dim_starts[ep])
        low_dim_ep_end = int(self.low_dim_ends[ep])

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
            elif source == "img":
                img_ep_start = int(self.img_starts[ep])
                img_ep_end = int(self.img_ends[ep])

                query_time = self.low_dim_replay_buffer[self.low_dim_time_key][query_idx]
                img_times = self.img_replay_buffer[self.img_time_key][img_ep_start:img_ep_end]
                # nearest image frame at or after the low_dim query time
                local_img_idx = np.searchsorted(np.squeeze(img_times), query_time)
                img_query_idx = int(
                    np.clip(img_ep_start + local_img_idx, img_ep_start, img_ep_end - 1)
                )

                arr = self.img_replay_buffer[key]
                obs_result[key] = self._slice_backward(
                    arr, img_query_idx, img_ep_start, horizon, down_sample_steps
                )
            else:
                raise ValueError(f"Unknown source '{source}' for obs key '{key}'")

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
