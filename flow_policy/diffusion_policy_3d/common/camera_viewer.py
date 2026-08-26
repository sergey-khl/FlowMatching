import time
import numpy as np
import cv2
from multiprocessing import Process, Event, shared_memory
from termcolor import cprint
from diffusion_policy_3d.common.precise_sleep import precise_wait

def _match_heights(frames):
    """hstack requires equal heights; resize any mismatched frame to the
    smallest common height, preserving aspect ratio."""
    heights = [f.shape[0] for f in frames]
    target_h = min(heights)
    out = []
    for f in frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = int(round(f.shape[1] * scale))
            f = cv2.resize(f, (new_w, target_h))
        out.append(f)
    return out


def _viewer_worker(view_configs, running_event, display_fps):
    """
    process code to see processed images
    """
    shms = {}
    arrays = {}
    last_seen = {name: 0 for name in view_configs}
    last_frame = {name: None for name in view_configs}

    try:
        for name, cfg in view_configs.items():
            shm = shared_memory.SharedMemory(name=cfg["shm_name"])
            shms[name] = shm
            arrays[name] = np.ndarray(cfg["shape"], dtype=np.dtype(cfg["dtype"]), buffer=shm.buf)

        window_name = "FLIR Cameras (press q to close viewer)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        dt = 1.0 / display_fps
        names_sorted = sorted(view_configs.keys())

        t_start = time.monotonic()
        iter_idx = 0



        while running_event.is_set():
            t_cycle_end = t_start + (iter_idx + 1) * dt

            for name in names_sorted:
                cfg = view_configs[name]
                with cfg["lock"]:
                    current_id = cfg["frame_id"].value
                    if current_id != 0 and current_id != last_seen[name]:
                        last_frame[name] = arrays[name].copy()
                        last_seen[name] = current_id

            frames_ready = [last_frame[n] for n in names_sorted]
            if all(f is not None for f in frames_ready):
                frames_ready = _match_heights(frames_ready)
                combined = np.hstack(frames_ready)
                cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running_event.clear()
                break

            precise_wait(t_cycle_end)
            iter_idx += 1

    except Exception as e:
        cprint(f"[CameraViewer] Error: {e}", "red")
    finally:
        cv2.destroyAllWindows()
        for shm in shms.values():
            shm.close()


class CameraViewer:
    """
    see flir cams in a seperate process
    """

    def __init__(self, multi_flir_manager, display_fps=30):
        self.multi_flir_manager = multi_flir_manager
        self.display_fps = display_fps
        self.running_event = Event()
        self.process = None

    def start(self):
        view_configs = {}
        for name, stream in self.multi_flir_manager.streams.items():
            view_configs[name] = {
                "shm_name": stream.shm.name,
                "shape": stream.shm_array.shape,
                "dtype": str(stream.shm_array.dtype),
                "frame_id": stream.frame_id,
                "lock": stream.lock,
            }

        self.running_event.set()
        self.process = Process(
            target=_viewer_worker,
            args=(view_configs, self.running_event, self.display_fps),
            daemon=True,
        )
        self.process.start()
        cprint("[CameraViewer] Viewer process started.", "green")

    def stop(self):
        self.running_event.clear()
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                cprint("[CameraViewer] WARNING: viewer did not exit, terminating.", "red")
                self.process.terminate()

