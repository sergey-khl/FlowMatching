from harvesters.core import Harvester
from termcolor import cprint
import time
from diffusion_policy_3d.common.flir_streamer import FLIRStream


class MultiFLIRManager:
    def __init__(
        self,
        camera_configs: dict,
        gentl_path: str = "/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
    ):
        """
        camera_configs: dict mapping camera names to serial numbers,
        e.g., {"wrist": "18475182", "front": "18475176"}
        """
        cprint("Initializing Harvester System...", "green")
        self.harvester = Harvester()
        self.harvester.add_cti_file(gentl_path)
        self.harvester.update_device_info_list()

        self.streams = {}
        for name, serial in camera_configs.items():
            self.streams[name] = FLIRStream(self.harvester, serial)

    def start_all(self):
        cprint("Starting all FLIR streams...", "green")
        for stream in self.streams.values():
            stream.start()
        # Give cameras a moment to warm up and fill buffers
        time.sleep(0.5)

    def read_all(self):
        """
        Reads from all cameras.
        Returns (True, {"wrist": img1, "front": img2}) if all succeeded.
        Returns (False, None) if ANY camera dropped a frame.
        """
        frames = {}
        for name, stream in self.streams.items():
            img = stream.read()
            if img is None:
                return False, None
            frames[name] = img

        return True, frames

    def stop_all(self):
        cprint("Shutting down FLIR streams...", "red")
        for stream in self.streams.values():
            stream.running = False
            stream.stop()
        self.harvester.reset()

