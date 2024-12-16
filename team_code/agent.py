import os
import datetime
import pathlib
import time
import cv2
import math
from collections import deque

import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms as T

import carla
from leaderboard.autoagents import autonomous_agent

from model import Net
from planner import RoutePlanner

import matplotlib.pyplot as plt

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'MyAgent'


class MyAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self._rgb_sensor_data = {"width": 400, "height": 300, "fov": 100}
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.frames_per_save = 20
        self.net = Net()

        self.last_rgb_f = deque(maxlen=10)
        self.forced_forward_steps = 0

        ckpt = torch.load(path_to_conf_file)
        self.net.load_state_dict(ckpt, strict=True)
        self.net.cuda()
        self.net.eval()

        self.save_path = None
        self._im_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_left').mkdir()
            (self.save_path / 'rgb_right').mkdir()
            (self.save_path / 'lidar').mkdir()
            (self.save_path / 'meta').mkdir()

    def _init(self):
        self._route_planner = RoutePlanner(4., 50.)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3, "y": 0.0,	"z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": -60.0,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": 60.0,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_right",
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            }
        ]

    def splat_points(self, point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 14
        y_meters_max = 28
        xbins = np.linspace(
            -2 * x_meters_max,
            2 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    def tick(self, input_data):
        self.step += 1

        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            'rgb_front': rgb_front,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'gps': gps,
            'speed': speed,
            'compass': compass,
        }

        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_wp'] = next_wp
        result['next_command'] = next_cmd.value

        theta = compass
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)

        result['target_point'] = tuple(local_command_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        
        tick_data = self.tick(input_data)
        speed = max(tick_data['speed'], 0.)
        
        rgb_front = self._im_transform(tick_data['rgb_front']).unsqueeze(0)
        rgb_left = self._im_transform(tick_data['rgb_left']).unsqueeze(0)
        rgb_right = self._im_transform(tick_data['rgb_right']).unsqueeze(0)

        if self.step < 1:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.0

            self.last_rgb_f.append(rgb_front)

            return control

        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
        
        speed = torch.FloatTensor([speed]).view(1, 1).to('cuda', dtype=torch.float32) / 20
        last_front = self.last_rgb_f[0].to('cuda', dtype=torch.float32)
        self.last_rgb_f.append(rgb_front)
        rgb_front = rgb_front.to('cuda', dtype=torch.float32)
        rgb_left = rgb_left.to('cuda', dtype=torch.float32)
        rgb_right = rgb_right.to('cuda', dtype=torch.float32)

        target_point = [torch.FloatTensor([tick_data['target_point'][0]]),
                        torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(target_point, dim=1).to('cuda', dtype=torch.float32) / 20
        state = torch.cat([speed, target_point, cmd_one_hot], 1)

        out1, out2, out3, _, _, _ = self.net(rgb_front, rgb_left, rgb_right, last_front, state)
        out2, out3 = out2.cpu().numpy(), out3.cpu().numpy()

        throttle, steer, brake = out3[0][0], out2[0][0], -out3[0][0]
        
        throttle = np.clip(throttle, 0, 0.75)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)

        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake

        if SAVE_PATH is not None and self.step % self.frames_per_save == 0:
            metadata = {
                'x' : float(tick_data['gps'][0]),
                'y' : float(tick_data['gps'][1]),
                'command' : int(command),
                'target_x' : float(tick_data['target_point'][0]),
                'target_y': float(tick_data['target_point'][1]),
                'speed' : float(speed),
                'predict_x' : float(out2[0][1]),
                'predict_y': float(out2[0][2]),
                'acc': float(out3[0][0]),
                'steer': float(out2[0][0]),
                'wp_x' : float(tick_data['next_wp'][0]),
                'wp_y': float(tick_data['next_wp'][1]),
            }
            size = 64
            out1 = out1.cpu().numpy().reshape((size, size))
            self.save(tick_data, metadata, out1)

        return control

    def save(self, tick_data, metadata, lidar=None):
        frame = self.step // self.frames_per_save
        Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))
        with open(self.save_path / 'meta' / ('%04d.json' % frame), 'w') as json_out:
            json.dump(metadata, json_out, indent=4)

        if lidar is not None:
            plt.imshow(lidar, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Value')
            plt.title('2D Array Heatmap')
            plt.savefig(self.save_path / 'lidar' / ('%04d.png' % frame), dpi=300, bbox_inches='tight') 
            plt.close()

    def destroy(self):
        del self.net
        torch.cuda.empty_cache()
