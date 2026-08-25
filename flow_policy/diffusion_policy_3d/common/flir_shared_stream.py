import ctypes
import time

import numpy as np
from multiprocessing import Process, Queue, Value, Lock, Event, shared_memory
from genicam.gentl import TimeoutException
from harvesters.core import Harvester
from termcolor import cprint


def _convert_frame(raw, width, height, data_format):
    """Runs in the child process only."""
    if "Mono" in data_format:
        return raw.reshape((height, width))
    elif "RGB" in data_format or "BGR" in data_format:
        channels = 4 if "a" in data_format.lower() else 3
        img = raw.reshape((height, width, channels))
        if "RGB" in data_format:
            img = img[:, :, :3][:, :, ::-1]
        return img
    elif "Bayer" in data_format:
        import cv2
        img_1d = raw.reshape((height, width))
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


def _flir_worker(serial, cti_path, startup_queue, frame_id, timestamp_ns,
                  consumed, lock, running_event, error_queue):
    """
    Owns the camera entirely in its own process. Only converts/publishes a
    new frame once the parent has consumed the previous one -- this avoids
    burning CPU on Bayer conversion for frames nobody ever reads, while also
    guaranteeing this work can never block the parent's control loop via the
    GIL, since it's a fully separate process.
    """
    device = None
    shm = None
    shm_array = None

    try:
        harvester = Harvester()
        harvester.add_cti_file(cti_path)
        harvester.update_device_info_list()
        device = harvester.create_image_acquirer(serial_number=str(serial))
        device.start_image_acquisition()

        while running_event.is_set():
            try:
                with device.fetch_buffer(timeout=0.5) as buffer:
                    if len(buffer.payload.components) == 0:
                        continue

                    with lock:
                        need_publish = consumed.value or (frame_id.value == 0)
                    if not need_publish:
                        continue  # parent hasn't read the last frame yet, skip conversion

                    component = buffer.payload.components[0]
                    raw = component.data.copy()
                    img = _convert_frame(raw, component.width, component.height,
                                          component.data_format)

                    if shm is None:
                        shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
                        shm_array = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
                        startup_queue.put({
                            "shm_name": shm.name,
                            "shape": img.shape,
                            "dtype": str(img.dtype),
                        })

                    with lock:
                        shm_array[:] = img
                        frame_id.value += 1
                        timestamp_ns.value = time.time_ns()
                        consumed.value = False

            except TimeoutException:
                continue
            except Exception as e:
                if not running_event.is_set():
                    break
                error_queue.put(str(e))
                time.sleep(0.05)

    except Exception as e:
        error_queue.put(f"Fatal init error on {serial}: {e}")
    finally:
        try:
            if device is not None:
                device.stop_image_acquisition()
                device.destroy()
        except Exception:
            pass
        if shm is not None:
            shm.close()
            shm.unlink()


class FLIRProcessStream:
    """
    Drop-in replacement for the thread-based FLIRStream. Same start()/read()/
    stop() interface, but capture + Bayer conversion happen in a dedicated
    OS process instead of a thread, so they can't starve the main loop's GIL.
    """

    def __init__(self, serial_num,
                 cti_path="/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti"):
        self.serial_num = serial_num
        self.cti_path = cti_path

        self.frame_id = Value(ctypes.c_uint64, 0)
        self.timestamp_ns = Value(ctypes.c_uint64, 0)
        self.consumed = Value(ctypes.c_bool, True)
        self.lock = Lock()
        self.running_event = Event()
        self.startup_queue = Queue()
        self.error_queue = Queue()

        self.process = None
        self.shm = None
        self.shm_array = None

    def start(self):
        self.running_event.set()
        self.process = Process(
            target=_flir_worker,
            args=(self.serial_num, self.cti_path, self.startup_queue,
                  self.frame_id, self.timestamp_ns, self.consumed,
                  self.lock, self.running_event, self.error_queue),
            daemon=True,
        )
        self.process.start()

        # Wait for the child to publish its shared-memory handle after its
        # first successfully converted frame.
        info = self.startup_queue.get(timeout=10.0)
        self.shm = shared_memory.SharedMemory(name=info["shm_name"])
        self.shm_array = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]),
                                     buffer=self.shm.buf)
        cprint(f"[FLIRProcessStream] {self.serial_num} attached to shm "
               f"'{info['shm_name']}' shape={info['shape']}", "green")

    def read(self, last_seen_id):
        with self.lock:
            current_id = self.frame_id.value
            if current_id == last_seen_id or current_id == 0:
                return None, last_seen_id, None
            img = self.shm_array.copy()
            ts = self.timestamp_ns.value
            self.consumed.value = True
        return img, current_id, ts

    def stop(self):
        self.running_event.clear()
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                cprint(f"WARNING: FLIR process {self.serial_num} did not exit, terminating.", "red")
                self.process.terminate()
        if self.shm is not None:
            self.shm.close()

