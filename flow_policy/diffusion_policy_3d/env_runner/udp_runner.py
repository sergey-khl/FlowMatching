import numpy as np
import torch
import collections
import tqdm
import time
import signal
import os
import struct
from termcolor import cprint
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import cv2

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

from diffusion_policy_3d.common.recorder_receiver import RecorderReceiver
from diffusion_policy_3d.common.multi_flir_manager import MultiFLIRManager
from diffusion_policy_3d.common.precise_sleep import precise_wait
from diffusion_policy_3d.common.udp_handler import TeleopUDPHandler
from diffusion_policy_3d.common.recorder import Recorder
import torchvision
 
class UdpRunner(BaseRunner):
    def __init__(self,
                 output_dir: str,
                remote_ip: str,
                follower_send_port: int,
                leader_send_port: int,
                recv_port: int,
                wrist_cam_serial: str,
                front_cam_serial: str,
                shape_meta: dict,
                n_action_steps: int,
                dof: int = 7,
                loop_hz: int = 10,
                debug: bool = False,
                ):
        super().__init__(output_dir)

        self.DOF = dof
        self.loop_hz = loop_hz
        self.debug = debug
        self.dt = 1/self.loop_hz

        self.n_action_steps = n_action_steps
        self.shape_meta = shape_meta
        self.obs_key_horizons = shape_meta['sample']['obs']['sparse']
        self.past_actions = None

        self.send_interval = 0.4  # after sending dont do anythign for at least this amount of time
        self.last_send_time = 0.0

        self.inference_time = 1.0          # estimate of how long inf takes
        self.inference_time_alpha = 0.2    # EMA smoothing factor for the estimate
 
        # UDP comms - same helper classes used in main.py
        self.listener = RecorderReceiver(remote_ip, recv_port, dof=self.DOF)
        self.follower_sender = TeleopUDPHandler(
            remote_ip, follower_send_port, action_size=8, DOF=self.DOF, horizon=self.n_action_steps
        )
        self.leader_sender = TeleopUDPHandler(
            remote_ip, leader_send_port, action_size=8, DOF=self.DOF, horizon=self.n_action_steps
        )

        camera_configs = {
            "wrist_img": wrist_cam_serial,
            "front_img": front_cam_serial,
        }
        self.camera_manager = MultiFLIRManager(camera_configs)
        self.camera_manager.start_all()

        self.loop_state = "IDLE"  # States: IDLE, INFER

        self.recorder = Recorder(save_dir="/mnt/10tb/serg/datasets/dataset")
        self.episode_counter = self.recorder.get_next_episode_index()

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()
 
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
 
        cprint("Initialization complete. Running UDP inference loop...", "green")
        cprint("Controls: [R]/o Start recording | [D]/x Stop+Save", "cyan")

    def _init_joystick(self):
        """Initializes the joystick file descriptor in non-blocking mode."""
        try:
            self.joy_fd = os.open("/dev/input/js0", os.O_RDONLY | os.O_NONBLOCK)
            cprint("[SYSTEM] Successfully opened joystick /dev/input/js0", "green")
        except OSError:
            cprint("[SYSTEM] Could not open joystick /dev/input/js0. Continuing without joystick.", "yellow")

    def _poll_joystick(self):
        """Reads non-blocking events from the joystick."""
        if self.joy_fd is None:
            return

        try:
            while True:
                event_data = os.read(self.joy_fd, 8)
                if not event_data:
                    break

                time_msec, value, ev_type, number = struct.unpack("IhBB", event_data)

                # Remove the init event flag (0x80)
                ev_type &= ~0x80

                if ev_type == 0x01 and value == 1:
                    # BLUETOOTHCTL connection
                    # if number == 1:  # 'o' button
                    #     print("press o")
                    #     self._handle_start_recording()
                    # elif number == 0:  # 'x' button
                    #     print("press x")
                    #     self._handle_stop_and_save()
                    # elif number == 8:   # hat Y axis, up = negative — confirm exact axis# and sign from your log!
                    #     print("press up")
                    #     self._handle_stop_and_save()
                    #     self._handle_select_dmp_demo()
                    # SIXAXIS connection
                    if number == 13:  # 'o' button
                        print("press o")
                        self._handle_start_recording()
                    elif number == 14:  # 'x' button
                        print("press x")
                        self._handle_stop_and_save()

        except BlockingIOError:
            pass
        except Exception as e:
            cprint(f"[SYSTEM] Error reading joystick: {e}", "red")


    def _on_key_press(self, key):
        """Asynchronous callback for keyboard events."""
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char.lower()

                if k == "r":
                    self._handle_start_recording()
                elif k == "d":
                    self._handle_stop_and_save()
        except Exception:
            pass

    def _handle_start_recording(self):
        """'r' on keyboard / 'o' on joystick: begin recording a new episode."""
        if self.loop_state == "IDLE":
            self.recorder.clear()
            self.loop_state = "INFER"
            self.last_send_time = 0.0
            cprint("\n[RUNNER] \U0001F534 EPISODE STARTED", "red", attrs=["bold"])

    def _handle_stop_and_save(self):
        """'d' on keyboard / 'x' on joystick: stop recording, save episode, stop any action sending."""
        if self.loop_state == "INFER":
            ep_name = f"episode_{self.episode_counter}"
            self.recorder.save_episode(ep_name)
            self.episode_counter += 1

            self.loop_state = "IDLE"

            cprint(f"\n[RUNNER] \U0001F4BE Saved {ep_name}. Ready for next episode.", "green")
            cprint("Press [R] or 'o' to start recording", "cyan")
    
    def _read_images(self):
        status, raw_frames = self.camera_manager.read_all()
        if not status:
            return False, None

        # TODO: we really shouldn't be doing this here, it would make more sense
        # for the policy to preprocess itself.
        processed_frames = {}
        for name, img_bgr in raw_frames.items():
            proc_img = preprocess_image(img_bgr)
            processed_frames[name] = proc_img

        return True, processed_frames

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.leader_sender.close()
        self.follower_sender.close()
        self.listener.close()
        if self.joy_fd is not None:
            try:
                os.close(self.joy_fd)
            except Exception:
                pass
        self.camera_manager.stop_all()

        os._exit(0)

    def _handle_new_frames(self, new_frames):
        for name, (img_bgr, ts_ns) in new_frames.items():
            proc = preprocess_image(img_bgr)
            proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            self.recorder.add_image_step(name, proc, ts_ns)

    def run(self, policy: BasePolicy):
        try:
            self.system_running = True

            def force_shutdown(signum, frame):
                cprint(
                    "\n[SYSTEM] Ctrl+C detected! Forcing main loop shutdown...", "red"
                )
                self.system_running = False

            signal.signal(signal.SIGINT, force_shutdown)

            policy = policy
            device = policy.device
            dtype = policy.dtype

            t_start = time.monotonic()
            iter_idx = 0
            while self.system_running:
                t_cycle_end = t_start + (iter_idx + 1) * self.dt

                self._poll_joystick()

                new_states = self.listener.receive_all_new()
                new_frames = self.camera_manager.read_all() # non blocking


                # run policy
                if self.loop_state == "INFER":
                    for state_dict in new_states:
                        self.recorder.add_low_dim_step(state_dict)
                    if new_frames:
                        self._handle_new_frames(new_frames)

                    # GET OBS_DICT HERE

                    time_to_chunk_end_s = time_to_chunk_end_ns / 1e9
                    interval_elapsed = (time.time() - self.last_send_time) >= self.send_interval

                    should_infer = (time_to_chunk_end_s <= self.inference_time) and interval_elapsed

                    if should_infer:
                        t_infer_start = time.monotonic()


                        with torch.no_grad():
                            action_dict = policy.predict_action(obs_dict, self.past_actions)

                        # device_transfer
                        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                        action_chunk = np_action_dict['action'].squeeze(0)
                        # action_chunk = np_action_dict['action_pred'].squeeze(0)

                        print("---------------------wrench")

                        for action in action_chunk:
                            print(list(action))
                        print("---------------------\n\n")

                        elapsed = time.monotonic() - t_infer_start
                        self.inference_time = (
                            self.inference_time_alpha * elapsed
                            + (1 - self.inference_time_alpha) * self.inference_time
                        )
                        print(f"took {self.inference_time}")
                        print(f"skipping {time_to_chunk_end_s}")

                        time_to_skip_ns = time_to_chunk_end_ns
                        self.follower_sender.send_action_chunk(action_chunk, time_to_skip_ns)
                        self.leader_sender.send_action_chunk(action_chunk, time_to_skip_ns)

                        self.last_send_time = time.time() # safety so we dont double infer on a similar state 

                precise_wait(t_cycle_end)
                iter_idx += 1

        except Exception as e:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()


