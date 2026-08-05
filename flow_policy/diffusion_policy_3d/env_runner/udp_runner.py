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

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

from diffusion_policy_3d.common.recorder_receiver import RecorderReceiver
from diffusion_policy_3d.common.multi_flir_manager import MultiFLIRManager
from diffusion_policy_3d.common.precise_sleep import precise_wait
from diffusion_policy_3d.common.udp_handler import TeleopUDPHandler
import torchvision

WAM_FORCE_LIMIT  = 5
WAM_TORQUE_LIMIT =  5
 
KP_JP = 100   # N/m
KP_POS = 100   # N/m
KP_ROT = 0   # N·m/rad
 

def quat_to_rotvec(quat):
    """compute state angle as a rotvec based on the home quat"""
    quat_xyzw = np.roll(quat, -1, axis=-1)
    ref_xyzw = np.roll(np.array([0.8311519102829725, 0.0005888003896006177, -0.556030721230029, -0.003999049322114239]), -1 , axis=-1) # home quat
    r = R.from_quat(ref_xyzw).inv() * R.from_quat(quat_xyzw)
    return r.as_rotvec()

def quat_to_euler(quats):
    """Quaternion [w, x, y, z] → Euler (roll, pitch, yaw) in radians."""
    quats_xyzw = np.roll(quats, -1, axis=-1)

    euler = R.from_quat(quats_xyzw).as_euler('xyz', degrees=False)  # (N,3)
    # return np.unwrap(euler, axis=0)
    return euler

def quat_to_quat(quat):
    """Quaternion [w, x, y, z] → quat"""
    ref_quat = np.array([0.8311519102829725, 0.0005888003896006177, -0.556030721230029, -0.003999049322114239]) # home quat
    if np.dot(quat, ref_quat) < 0:
        quat *= -1
    return quat

def quat_to_rotmat(quat):
    """Quaternion [w, x, y, z] -> first two rows of the rotation matrix."""
    quat_xyzw = np.roll(quat, -1, axis=-1)
    rotmat = R.from_quat(quat_xyzw).as_matrix()
    return rotmat[:2, :].reshape(6)

def preprocess_image(image):
    img_size = 84

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def _rotvec_delta_to_rot_error(delta_rotvec: np.ndarray) -> np.ndarray:
    """
    Convert an incremental euler-XYZ delta (radians, EE frame) to a rotation
    error vector suitable for a proportional torque controller.
    """
    q = R.from_rotvec(delta_rotvec).as_quat()
    qx, qy, qz, qw = q
    return 2.0 * np.sign(qw) * np.array([qx, qy, qz])
 
 
class UdpRunner(BaseRunner):
    def __init__(self,
                 output_dir: str,
                remote_ip: str,
                follower_send_port: int,
                leader_send_port: int,
                recv_port: int,
                wrist_cam_serial: str,
                front_cam_serial: str,
                n_action_steps: int,
                n_obs_steps: int,
                horizon: int,
                dof: int = 7,
                loop_hz: int = 10,
                debug: bool = False,
                ):
        super().__init__(output_dir)

        self.DOF = dof
        self.loop_hz = loop_hz
        self.debug = debug
        self.dt = 1/self.loop_hz

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.obs_history = collections.deque(maxlen=self.n_obs_steps)
        self.past_actions = None

        self.send_interval = 0.1  # after sending dont do anythign for at least this amount of time
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

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()
 
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
 
        cprint("Initialization complete. Running UDP inference loop...", "green")
        cprint("Controls: [R] Start/Stop inference", "cyan")

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

                time_msec, value, ev_type, number = struct.unpack('IhBB', event_data)

                # Remove the init event flag (0x80)
                ev_type &= ~0x80

                # ev_type == 1 means button event, value == 1 means pressed (down)
                if ev_type == 0x01 and value == 1:
                    if number == 1:  # 'o' button
                        self._handle_start_save_action()
                    elif number == 0:  # 'x' button
                        print("press x")
                        self._handle_stop()

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
                    self._handle_start_save_action()
                elif k == "d":
                    self._handle_stop()
        except Exception:
            pass

    def _handle_start_save_action(self):
        """Unified logic for 'r' / 's' on keyboard and 'o' on joystick."""
        if self.loop_state == "IDLE":
            self.loop_state = "INFER"
            self.obs_history.clear()
            cprint("\n[INFERENCE] 🔴 EPISODE STARTED", "red", attrs=["bold"])

    def _handle_stop(self):
        """'d' on keyboard / 'x' on joystick: stop any action sending."""
        if self.loop_state == "INFER":
            self.loop_state = "IDLE"
            self.obs_history.clear()

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

    def _read_state(self):
        state_dict = self.listener.receive_latest_data()
        status = state_dict is not None

        return status, state_dict

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.camera_manager.stop_all()
        self.follower_sender.close()
        if self.joy_fd is not None:
            try:
                os.close(self.joy_fd)
            except Exception:
                pass

        os._exit(0)


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

            waiting_counter = 0
            t_start = time.monotonic()
            iter_idx = 0
            while self.system_running:
                t_cycle_end = t_start + (iter_idx + 1) * self.dt

                self._poll_joystick()

                # Read images
                img_status, image_dict = self._read_images()
                state_status, state_dict = self._read_state()

                if not state_status:
                    if waiting_counter == 10:
                        cprint("waiting for wam", "yellow")
                        waiting_counter = 0
                    waiting_counter += 1
                    continue
                if not img_status:
                    if waiting_counter == 10:
                        cprint("waiting for img", "yellow")
                        waiting_counter = 0
                    waiting_counter += 1
                    continue

                obs = image_dict.copy()

                ee = state_dict["follower_cart_pos"]
                jp = state_dict["follower_jp"]
                # position = state_dict["follower_jp"]
                rotation = quat_to_rotmat(state_dict["follower_quat_wxyz"])
                # rotation = quat_to_quat(state_dict["follower_quat_wxyz"])
                gripper = state_dict["gripper_pos"]
                time_to_chunk_end_ns = state_dict["time_to_chunk_end_ns"]
                timestamp_ns = state_dict["timestamp_ns"]


                state = np.concatenate([
                    jp,
                    # ee,
                    # rotation,
                    [gripper],
                    # [-gripper],
                ]).astype(np.float32)  # (8,)

                obs["agent_pos"] = state

                self.obs_history.append(obs)

                print("---------------------state")
                print(list(state))
                if len(self.obs_history) < self.n_obs_steps:
                    continue

                # stack along a new time axis: each key -> (n_obs_steps, ...)
                stacked_obs = {
                    key: np.stack([o[key] for o in self.obs_history], axis=0)
                    for key in obs.keys()
                }

                # device transfer
                obs_dict = dict_apply(stacked_obs,
                                      lambda x: torch.from_numpy(x).unsqueeze(0).to(
                                          device=device))

                # run policy
                if self.loop_state == "INFER":
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

                        self.obs_history.clear() # fill with new data

                        self.last_send_time = time.time() # safety so we dont double infer on a similar state 

                precise_wait(t_cycle_end)
                iter_idx += 1

        except Exception as e:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()


