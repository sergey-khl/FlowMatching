import cv2
import numpy as np
import threading
import time
from genicam.gentl import TimeoutException
from harvesters.core import Harvester
from termcolor import cprint


def buffer_to_image(buffer):
    component = buffer.payload.components[0]
    width = component.width
    height = component.height
    data_format = component.data_format

    raw_data = component.data.copy()

    if "Mono" in data_format:
        return raw_data.reshape((height, width))

    elif "RGB" in data_format or "BGR" in data_format:
        channels = 4 if "a" in data_format.lower() else 3
        img_nd = raw_data.reshape((height, width, channels))
        if "RGB" in data_format:
            img_nd = img_nd[:, :, :3][:, :, ::-1]
        return img_nd

    elif "Bayer" in data_format:
        img_1d = raw_data.reshape((height, width))
        if "BayerRG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)
        elif "BayerBG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerRG2BGR)
        elif "BayerGB" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGR2BGR)
        elif "BayerGR" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGB2BGR)
        else:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)

    else:
        raise ValueError(f"Can't convert {data_format}")


class FLIRStream:
    def __init__(self, harvester: Harvester, serial_num: str = None):
        self.harvester = harvester
        self.serial_num = serial_num

        cprint(
            f"Opening FLIR camera {serial_num if serial_num else 'default'}...", "cyan"
        )
        if self.serial_num:
            self.device = self.harvester.create_image_acquirer(serial_number=str(self.serial_num))
        else:
            self.device = self.harvester.create_image_acquirer()

        self.latest_image = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.running = True
        self.device.start_image_acquisition()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                with self.device.fetch_buffer(timeout=0.5) as buffer:
                    if len(buffer.payload.components) == 0:
                        continue

                    img = buffer_to_image(buffer)
                    with self.lock:
                        self.latest_image = img
            except TimeoutException:
                continue
            except Exception as e:
                if not self.running:
                    break
                else:
                    cprint(
                        f"Unexpected FLIR stream error on {self.serial_num}: {e}", "red"
                    )
                    time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.latest_image is not None:
                return self.latest_image.copy()
            return None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.device.stop_image_acquisition()
            # CRITICAL: Clean up the GenTL node so the camera isn't locked on next run
            self.device.destroy()
        except Exception as e:
            cprint(f"Error closing FLIR camera {self.serial_num}: {e}", "red")
