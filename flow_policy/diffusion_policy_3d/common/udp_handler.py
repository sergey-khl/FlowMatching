import socket
import struct
import time
import numpy as np


class TeleopUDPHandler:
    def __init__(self, remote_ip, send_port, action_size=7, DOF=7, horizon=5):
        """
        :param remote_ip: IP address of the target (e.g., '127.0.0.1')
        :param send_port: The port the target is listening on.
        :param dof: Degrees of freedom (default 7).
        """
        self.dof = DOF
        self.horizon = horizon

        self.remote_ip = remote_ip
        self.send_port = send_port

        # self.action_size = 7
        self.action_size = action_size
        
        # jp (DOF doubles) + jv (DOF doubles) + ext_torque (DOF doubles) +
        # meas_torque (DOF doubles) + gripper (1 double) + timestamp (uint64_t)
        self.header_fmt = "<Q"
        self.packet_size = struct.calcsize(self.header_fmt) + (
            self.horizon * self.action_size * 8
        )

        # 1. Sender Socket (Always created)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self.sock_send.setblocking(False)

        print(f"UDP [inference send]: {remote_ip}:{send_port}.")


    def send_action_chunk(self, actions, timestamp_ns):
        """
        Sends a chunk of actions to the remote target in one packet.
        """
        actions = np.asarray(actions[:self.horizon, :], dtype="<f8")
        if actions.shape != (self.horizon, self.action_size):
            raise ValueError(
                f"Expected shape ({self.horizon}, {self.action_size}), "
                f"got {actions.shape}"
            )

        header = struct.pack(self.header_fmt, timestamp_ns)
        payload = header + actions.tobytes()

        try:
            self.sock_send.sendto(payload, (self.remote_ip, self.send_port))
        except BlockingIOError:
            # OS send buffer full; drop this chunk.
            pass
        except Exception as e:
            print(f"Send Error: {e}")


    def close(self):
        if self.sock_send: 
            self.sock_send.close()

