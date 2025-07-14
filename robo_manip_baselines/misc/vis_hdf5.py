#!/usr/bin/env python

import argparse
import io
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import h5py

from robo_manip_baselines.common import convert_depth_image_to_point_cloud, crop_and_resize


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("teleop_filename", type=str)
    parser.add_argument("--skip", default=10, type=int, help="skip frames")
    parser.add_argument(
        "-o", "--output_mp4_filename",
        type=str,
        help="save result as mp4 file when this option is set"
    )
    parser.add_argument(
        "--mp4_codec",
        type=str,
        default="mp4v",
        help="fourcc codec for mp4"
    )
    parser.add_argument(
        "--rgb_crop_size_list",
        type=int,
        nargs='+',
        default=None,
        help="List of rgb crop sizes: either [w,h] or [w1,h1,w2,h2,...]"
    )
    return parser.parse_args()


def load_raw_episode_data(ep_path: Path):
    """
    Returns:
      imgs_per_cam: dict[str, np.ndarray] of shape (T,H,W,3)
      state:    np.ndarray of shape (T, J)
      action:   np.ndarray of shape (T, J)
      vel:      np.ndarray or None
      effort:   np.ndarray or None
    """
    with h5py.File(ep_path, 'r') as ep:
        state = ep['/measured_joint_pos'][::3]
        action = ep['/command_joint_pos'][::3]
        vel = ep['/measured_joint_vel'][::3] if '/measured_joint_vel' in ep else None
        effort = ep['/effort'][::3] if '/effort' in ep else None

        # collect all RGB cameras
        imgs_per_cam = {}
        for key in ep.keys():
            if key.endswith('_rgb_image'):
                cam_name = key.strip('/')[:-len('_image')]
                imgs_per_cam[cam_name] = ep[f'/{key}'][::3]

    return imgs_per_cam, state, action, vel, effort


class VisualizeData:
    def __init__(
        self, teleop_filename, skip, output_mp4_filename, mp4_codec, rgb_crop_size_list
    ):
        self.data_setup(Path(teleop_filename), skip, rgb_crop_size_list)
        self.figure_axes_setup()
        self.video_writer_setup(output_mp4_filename, mp4_codec)
        self.axes_limits_configuration()
        self.plot_lists_initialization()

    def data_setup(self, ep_path, skip, rgb_crop_size_list):
        print(f"Loading HDF5 episode from {ep_path}...")
        imgs_per_cam, state, action, vel, effort = load_raw_episode_data(ep_path)
        self.imgs_per_cam = imgs_per_cam
        self.state      = state
        self.action     = action
        self.velocity   = vel
        self.effort     = effort
        self.sensor_names = list(imgs_per_cam.keys())
        self.skip = skip

        # optional crop sizes
        if rgb_crop_size_list is None:
            self.rgb_crop_size_list = None
        else:
            def refine(lst):
                if len(lst)==2:
                    return [tuple(lst)] * len(self.sensor_names)
                assert len(lst)==len(self.sensor_names)*2
                return [ (lst[i], lst[i+1]) for i in range(0,len(lst),2) ]
            self.rgb_crop_size_list = refine(rgb_crop_size_list)

        # synthetic time axis
        self.time_array = np.arange(self.state.shape[0])

    def figure_axes_setup(self):
        plt.rcParams['keymap.quit'] = ['q','escape']
        n = len(self.sensor_names)
        self.fig, self.ax = plt.subplots(n+1,4, figsize=(16,12), constrained_layout=True)
        # convert last two in each row to 3D for sensors
        for i in range(1,n+1):
            # remove old depth & point axes
            if self.ax[i,1] in self.fig.axes:
                self.ax[i,1].remove()
            if self.ax[i,2] in self.fig.axes:
                self.ax[i,2].remove()
            # reassign 3D subplot
            self.ax[i,2] = self.fig.add_subplot((n+1),4,4*(i+1)-1, projection='3d')
        self.break_flag = False

    def video_writer_setup(self, out_fn, codec):
        self.video_writer = None
        if out_fn:
            base,ext = os.path.splitext(out_fn)
            if ext.lower()!='.mp4': out_fn = base+'.mp4'
            os.makedirs(os.path.dirname(out_fn) or '.', exist_ok=True)
            w = int(self.fig.get_figwidth()*self.fig.dpi)
            h = int(self.fig.get_figheight()*self.fig.dpi)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(out_fn, fourcc, 10, (w,h))

    def axes_limits_configuration(self):
        t0,t1 = self.time_array[0], self.time_array[-1]
        # joint pos
        self.ax[0,0].set_title('joint pos'); self.ax[0,0].set_xlim(t0,t1)
        # joint vel vs effort
        self.ax[0,1].set_title('joint vel'); self.ax[0,1].set_xlim(t0,t1)
        self.ax[0,2].set_title('eef pose'); self.ax[0,2].set_xlim(t0,t1)
        self.ax[0,3].set_title('eef wrench'); self.ax[0,3].set_xlim(t0,t1)

        # dynamic y-limits
        self.ax[0,0].set_ylim(
            np.min(self.action[:,:-1]), np.max(self.action[:,:-1])
        )
        ax00_twin = self.ax[0,0].twinx()
        ax00_twin.set_ylim(
            np.min(self.action[:,-1]), np.max(self.action[:,-1])
        )
        self.ax00_twin = ax00_twin

        vel = self.velocity if self.velocity is not None else np.zeros_like(self.state)
        self.ax[0,1].set_ylim(np.min(vel[:,:-1]), np.max(vel[:,:-1]))

        pose = np.concatenate([self.state[:,:3], self.state[:,3:]],axis=1)
        self.ax[0,2].set_ylim(np.min(pose[:,:3]), np.max(pose[:,:3]))
        ax02_twin = self.ax[0,2].twinx(); ax02_twin.set_ylim(-1,1)
        self.ax02_twin = ax02_twin

    def plot_lists_initialization(self):
        self.data_list = {'time':[], 'cmd_pos':[], 'meas_pos':[], 'meas_vel':[]}

    def clear_axis(self, ax):
        for c in ax.get_children():
            if isinstance(c, plt.Line2D): c.remove()
        ax.set_prop_cycle(None)

    def handle_rgb_image(self, sensor_idx, t):
        ax = self.ax[sensor_idx+1,0]
        ax.axis('off')
        img = self.imgs_per_cam[self.sensor_names[sensor_idx]][t]
        if self.rgb_crop_size_list:
            img = crop_and_resize(img[np.newaxis], crop_size=self.rgb_crop_size_list[sensor_idx])[0]
        ax.imshow(img[::4,::4])

    def handle_depth_image(self, sensor_idx, t):
        # no depth in HDF5: ensure axis removed only once
        ax = self.ax[sensor_idx+1,1]
        if ax in self.fig.axes:
            ax.remove()

    def plot(self):
        T = len(self.time_array)
        for t in tqdm(range(0,T,self.skip)):
            if self.break_flag: break

            times = self.time_array[:t+1]
            cmd = self.action[:t+1]
            meas = self.state[:t+1]
            vel  = self.velocity[:t+1] if self.velocity is not None else None

            # joint pos
            self.clear_axis(self.ax[0,0]); self.ax[0,0].plot(times, cmd[:,:-1],'--'); self.ax[0,0].plot(times, meas[:,:-1])
            # gripper
            self.ax00_twin.plot(times, cmd[:,-1],'--'); self.ax00_twin.plot(times, meas[:,-1])

            # joint vel
            self.clear_axis(self.ax[0,1]);
            if vel is not None: self.ax[0,1].plot(times, vel[:,:-1])

            # sensor images
            for i in range(len(self.sensor_names)):
                self.handle_rgb_image(i,t)
                self.handle_depth_image(i,t)

            plt.draw(); plt.pause(0.001)
            if self.video_writer:
                buf = io.BytesIO()
                self.fig.savefig(buf,format='jpg'); buf.seek(0)
                arr = np.frombuffer(buf.read(),dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                self.video_writer.write(frame)

            self.fig.canvas.mpl_connect('key_press_event', self.key_event)

        if self.video_writer:
            self.video_writer.release()

        print("Press 'Q' or 'Esc' to quit.")
        plt.show()

    def key_event(self, event):
        if event.key in ['q','escape']:
            self.break_flag = True


if __name__ == '__main__':
    args = parse_argument()
    viz = VisualizeData(
        args.teleop_filename,
        args.skip,
        args.output_mp4_filename,
        args.mp4_codec,
        args.rgb_crop_size_list,
    )
    viz.plot()
