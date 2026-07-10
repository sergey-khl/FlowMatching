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
from diffusion_policy_3d.common.udp_handler import TeleopUDPHandler
import torchvision

WAM_FORCE_LIMIT  = 5
WAM_TORQUE_LIMIT =  5
 
KP_POS = 300   # N/m
KP_ROT = 5   # N·m/rad
 

def quat_to_rotvec(quat):
    """compute state angle as a rotvec based on the home quat"""
    quat_xyzw = np.roll(quat, -1, axis=-1)
    ref_xyzw = np.roll(np.array([0.8311519102829725, 0.0005888003896006177, -0.556030721230029, -0.003999049322114239]), -1 , axis=-1) # home quat
    r = R.from_quat(ref_xyzw).inv() * R.from_quat(quat_xyzw)
    return r.as_rotvec()

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
 
 
def delta_to_wrench(delta: np.ndarray) -> np.ndarray:
    """
    Convert a single incremental delta [dx, dy, dz, droll, dpitch, dyaw, gripper]
    expressed in the EE frame into a wrench [fx, fy, fz, tx, ty, tz, gripper].
    """
    delta_pos        = delta[:3]
    delta_rotvec  = delta[3:6]
    gripper          = delta[6]
 
    force  = KP_POS * delta_pos
    torque = KP_ROT * _rotvec_delta_to_rot_error(delta_rotvec)
 
    force  = np.clip(force,  -WAM_FORCE_LIMIT,  WAM_FORCE_LIMIT)
    torque = np.clip(torque, -WAM_TORQUE_LIMIT, WAM_TORQUE_LIMIT)
 
    return np.concatenate([force, torque, [gripper]])
 



class UdpRunner(BaseRunner):
    def __init__(self,
                 output_dir: str,
                remote_ip: str,
                send_port: int,
                recv_port: int,
                wrist_cam_serial: str,
                front_cam_serial: str,
                dof: int = 7,
                action_horizon: int = 5,
                loop_hz: int = 10,
                debug: bool = False,
                n_obs_steps: int = 2,
                ):
        super().__init__(output_dir)

        self.DOF = dof
        self.action_horizon = action_horizon
        self.loop_hz = loop_hz
        self.debug = debug

        self.n_obs_steps = n_obs_steps
        self.obs_history = collections.deque(maxlen=self.n_obs_steps)
 
        # UDP comms - same helper classes used in main.py
        self.follower_listener = RecorderReceiver(remote_ip, recv_port, dof=self.DOF)
        self.follower_sender = TeleopUDPHandler(
            remote_ip, send_port, DOF=self.DOF, horizon=self.action_horizon
        )

        camera_configs = {
            "wrist_img": wrist_cam_serial,
            "front_img": front_cam_serial,
        }
        self.camera_manager = MultiFLIRManager(camera_configs)
        self.camera_manager.start_all()

        # simple on/off toggle for sending actions - no recording states needed
        self.doing_inference = False
        
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
                    if self.loop_state == "IDLE":
                        self._handle_start_save_action()
        except Exception:
            pass

    def _handle_start_save_action(self):
        """Unified logic for 'r' / 's' on keyboard and 'o' on joystick."""
        if self.loop_state == "IDLE":
            self.loop_state = "INFER"
            self.doing_inference = True
            self.obs_history.clear()
            cprint("\n[INFERENCE] 🔴 EPISODE STARTED", "red", attrs=["bold"])
        elif self.loop_state == "INFER":
            self.loop_state = "IDLE"
            self.doing_inference = False
            cprint("\n[INFERENCE] ⏸ EPISODE PAUSED", "yellow")
    
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
        state_dict = self.follower_listener.receive_latest_data()
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

            loop_delay = 1 / self.loop_hz
            waiting_counter = 0
            while self.system_running:
                loop_start_time = time.time()

                self._poll_joystick()

                # Read images
                img_status, image_dict = self._read_images()
                state_status, state_dict = self._read_state()

                if not state_status:
                    if waiting_counter == 10:
                        cprint("waiting for messages", "yellow")
                        waiting_counter = 0
                    waiting_counter += 1
                    continue

                obs = image_dict.copy()

                position = state_dict["follower_cart_pos"]
                rotation = quat_to_rotvec(state_dict["follower_cart_rot"])
                gripper = state_dict["gripper_pos"]
                
                state = np.concatenate([
                    position,
                    rotation,
                    [gripper],
                    [-gripper],
                ]).astype(np.float32)  # (8,)

                obs["agent_pos"] = state

                self.obs_history.append(obs)

                while len(self.obs_history) < self.n_obs_steps:
                    self.obs_history.append(obs)

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
                if self.doing_inference and self.loop_state == "INFER":
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                    action_chunk = np_action_dict['action'].squeeze(0)

                    wrench_chunk = np.stack(
                        [delta_to_wrench(d) for d in action_chunk]
                    )

                    print(wrench_chunk)

                    self.follower_sender.send_action_chunk(wrench_chunk)

                # sleep, accounting for the time the loop already took to try and keep the desired frequency
                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0.0, loop_delay - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()


