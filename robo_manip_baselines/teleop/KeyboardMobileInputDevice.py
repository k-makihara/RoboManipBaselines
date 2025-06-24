import threading

import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class KeyboardMobileInputDevice(InputDeviceBase):
    """Keyboard for teleoperation input device."""

    def __init__(
        self,
        mobile_manager,
        xy_scale=0.2,
        theta_scale=0.2,
    ):
        super().__init__()

        self.mobile_manager = mobile_manager
        self.xy_scale = xy_scale
        self.theta_scale = theta_scale

        self.state = {
            # position control keys
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "q": False,
            "e": False,
        }

        self.listener = None
        self.listener_thread = None

    def connect(self):
        if self.connected:
            return

        from pynput import keyboard

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        # keyboard listener another thread
        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.connected = True
        print(f"[{self.__class__.__name__}] Connected.")
        print(f"""[{self.__class__.__name__}] Key Bindings:
            - WASD : XY movement
            - QE   : Z-axis rotation""")

    def _start_listener(self):
        self.listener.start()
        self.listener.join()

    def _on_press(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = True
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = False
        except AttributeError:
            pass

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

    def set_command_data(self):
        delta_pos = np.zeros(3)

        # X-axis
        if self.state["w"]:
            delta_pos[0] += self.xy_scale
        if self.state["s"]:
            delta_pos[0] -= self.xy_scale

        # Y-axis
        if self.state["a"]:
            delta_pos[1] += self.xy_scale
        if self.state["d"]:
            delta_pos[1] -= self.xy_scale

        # Z-axis
        if self.state["q"]:
            delta_pos[2] += self.theta_scale
        if self.state["e"]:
            delta_pos[2] -= self.theta_scale

        

        vel = np.array(
            [
                delta_pos[0],
                delta_pos[1],
                2.0 * delta_pos[2],
            ]
        )

        self.mobile_manager.set_command_vel(vel)

    def disconnect(self):
        if self.connected:
            if self.listener:
                self.listener.stop()
            self.connected = False
            print(f"[{self.__class__.__name__}] Disconnected.")
