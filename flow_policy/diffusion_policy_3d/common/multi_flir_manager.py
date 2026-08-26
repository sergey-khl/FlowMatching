from termcolor import cprint
import time
from diffusion_policy_3d.common.flir_shared_stream import FLIRProcessStream


class MultiFLIRManager:
    def __init__(
        self,
        camera_configs: dict,
        gentl_path: str = "/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
    ):
        cprint("Initializing FLIR camera processes...", "green")
        self.streams = {}
        self.last_ids = {}
        for name, cfg in camera_configs.items():
            serial = cfg["serial"]
            camera_settings = {k: v for k, v in cfg.items() if k != "serial"}
            self.streams[name] = FLIRProcessStream(serial, gentl_path, camera_settings=camera_settings)
            self.last_ids[name] = 0

    def start_all(self):
        cprint("Starting all FLIR camera processes...", "green")
        for stream in self.streams.values():
            stream.start()
        time.sleep(0.5)

    def read_all(self):
        new_frames = {}
        for name, stream in self.streams.items():
            img, new_id, ts_ns = stream.read(self.last_ids[name])
            if img is not None:
                self.last_ids[name] = new_id
                new_frames[name] = (img, ts_ns)
        return new_frames

    def stop_all(self):
        cprint("Shutting down FLIR camera processes...", "red")
        for stream in self.streams.values():
            stream.stop()

