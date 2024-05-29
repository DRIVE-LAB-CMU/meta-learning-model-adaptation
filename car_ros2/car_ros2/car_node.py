import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker


import numpy as np
import matplotlib.pyplot as plt

from tf_transformations import quaternion_from_euler, euler_matrix

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_jax import MPPIController, MPPIParams, rollout_fn_select
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
from std_msgs.msg import Float64
import torch.nn as nn
import torch.optim as optim
import torch
import time
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
print("DEVICE", jax.devices())

DT = .05
N_ROLLOUTS = 10000
H = 8
SIGMA = 1.0
LF = .16
LR = .15
L = LF+LR

trajectory_type = "counter oval"

SPEED = 1.8

model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=0.36, Sb=0.0,Ta=20., Tb=.0, mu=0.5)


dynamics = DynamicBicycleModel(model_params)


model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.36, Sb=-0., Ta=20., Tb=.0, mu=0.5)

dynamics_single = DynamicBicycleModel(model_params_single)
env =OffroadCar({}, dynamics_single)

rollout_fn = rollout_fn_select('dbm', dynamics, DT, L, LR)

dynamics.reset()
dynamics_single.reset()

def fn():
    ...

mppi_params = MPPIParams(
        sigma = 1.0,
        gamma_sigma=.0,
        gamma_mean = 1.0,
        discount=1.,
        sample_sigma = 1.,
        lam = 0.01,
        n_rollouts=N_ROLLOUTS,
        H=H,
        a_min = [-1., -1.],
        a_max = [1., 1.],
        a_mag = [1., 1.], # 0.1, 0.35
        a_shift = [.0, 0.],
        delay=0,
        len_history=2,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        smooth_alpha=1.,
)

mppi = MPPIController(
    mppi_params, rollout_fn, fn, key
)

waypoint_generator = WaypointGenerator(trajectory_type, DT, H, SPEED)
done = False
frames = []

obs = env.reset()


        
goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []


pos2d = []
target_list_all = []

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.fc2.bias.data = torch.tensor([0.,0.,0.])

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleModel(5,100,3)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=.02)
model = model.double()

def rollout_nn(state,actions) :
    global model
    traj = []
    state = torch.tensor(state).double()
    for action in actions :
        x = state[0]
        y = state[1]
        psi = state[2]
        vx = state[3]   
        vy = state[4]
        omega = state[5]
        action = torch.tensor(action).double()
        X = torch.cat((state[3:].unsqueeze(0),action.unsqueeze(0)),dim=1)
        # print(model(X))
        state[3] += model(X).detach()[0,0]*DT
        state[4] += model(X).detach()[0,1]*DT
        state[5] += model(X).detach()[0,2]*DT
        state[0] += state[3]*np.cos(state[2])*DT - state[4]*np.sin(state[2])*DT
        state[1] += state[3]*np.sin(state[2])*DT + state[4]*np.cos(state[2])*DT
        state[2] += state[5]*DT
        # print(state)
        traj.append([float(state[0]),float(state[1]),float(state[2])])
    # print(np.array(traj))
    return np.array(traj)

def train_step(states,cmds,augment=True) :
    global model
    X = np.concatenate((np.array(states),np.array(cmds)),axis=1)[:-1]
    Y = (np.array(states)[1:] - np.array(states)[:-1])
    # print(X,Y)
    
    # Augmentation
    if augment :
        X_ = X.copy()
        X_[:,1] = -X[:,1]
        X_[:,2] = -X[:,2]
        X_[:,4] = -X[:,4]
        Y_ = Y.copy()
        Y_[:,1] = -Y[:,1]
        Y_[:,2] = -Y[:,2]
        
        X = np.concatenate((X,X_),axis=0)    
        Y = np.concatenate((Y,Y_),axis=0)
    
        X_ = X.copy()
        X_[:,1] = 0.
        X_[:,2] = 0.
        X_[:,4] = 0.
        Y_ = Y.copy()
        Y_[:,1] = 0.
        Y_[:,2] = 0.
        
        X = np.concatenate((X,X_),axis=0)
        Y = np.concatenate((Y,Y_),axis=0)
    
    X = torch.tensor(X).double()
    X[:,3] = 0.
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)*DT
    
    Y = torch.tensor(Y)
    # Compute the loss
    loss = criterion(outputs/DT, Y/DT)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()    
    
    return


class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.path_pub_nn = self.create_publisher(Path, 'path_nn', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        self.odom_pub_ = self.create_publisher(Odometry, 'odom', 1)
        self.timer_ = self.create_timer(0.05, self.timer_callback)
        self.slow_timer_ = self.create_timer(1.0, self.slow_timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'throttle', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steer', 1)
        self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.states = []
        self.cmds = []
        
    def timer_callback(self):
        global obs, target_list_all
        # distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
        # # import pdb; pdb.set_trace()
        # t_idx = np.argmin(distance_list)
        # t_closed = waypoint_t_list[t_idx]
        # target_pos_list = [reference_traj(0. + t_closed + i*DT*1.) for i in range(H+0+1)]
        # target_pos_tensor = jnp.array(target_pos_list)
        target_pos_tensor = waypoint_generator.generate(jnp.array(obs[:5]))
        target_pos_list = np.array(target_pos_tensor)
        dynamics.reset()
        # print(env.obs_state())
        
        # target_list_all += target_pos_list
        # action, mppi_info = mppi(obs, reward_fn(target_pos_tensor))
        # print("obs", env.obs_state())
        t1 = time.time()
        action, mppi_info = mppi(env.obs_state(),target_pos_tensor, vis_optim_traj=True)
        t2 = time.time()
        print("Time taken", t2-t1)
        action = np.array(action)
        # action[1] = action[1] * 1.5
        sampled_traj = np.array(mppi_info['trajectory'][:, :2])
        # sampled_traj = np.zeros((H+1, 2))
        # print(action)
        # import pdb; pdb.set_trace()
        # all_trajectory = np.array(mppi_info['all_traj'])[:, :, :2]
        # all_trajectory = np.zeros((H+1, 1000, 2))
        # plt.figure()
        # for i in range(10):
        #     plt.plot(all_trajectory[:, i, 0], all_trajectory[:, i, 1])
        # plt.savefig('all_trajectory.png')
        # print(all_trajectory.shape)
        # action = np.zeros(2)
        # action *= 0.
        # action = np.array([.1, 1.])
        px, py, psi, vx, vy, omega = env.obs_state().tolist()
        self.states.append([vx,vy,omega])
        obs, reward, done, info = env.step(action)
        # print("new obs", env.obs_state())
        
        # pos2d.append(env.obs_state()[:2])
        # obs_list.append(env.obs_state()) 
        
        
        q = quaternion_from_euler(0, 0, psi)
        now = self.get_clock().now().to_msg()
        
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = now
        pose.pose.pose.position.x = px
        pose.pose.pose.position.y = py
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]
        self.pose_pub_.publish(pose)
        
        
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = now
        for i in range(target_pos_list.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(target_pos_list[i][0])
            pose.pose.position.y = float(target_pos_list[i][1])
            path.poses.append(pose)
        self.ref_trajectory_pub_.publish(path)
        
        mppi_path = Path()
        mppi_path.header.frame_id = 'map'
        mppi_path.header.stamp = now
        for i in range(len(sampled_traj)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(sampled_traj[i, 0])
            pose.pose.position.y = float(sampled_traj[i, 1])
            mppi_path.poses.append(pose)
        self.path_pub_.publish(mppi_path)
        
        w_pred_ = 0.
        _w_pred = 0.
        self.cmds.append([0., action[1]])
        if len(self.states) > 100  :
            self.states = self.states[-100:]
            self.cmds = self.cmds[-100:]
            train_step(self.states,self.cmds,augment = True)
            
            X = np.concatenate((np.array(self.states[-1:]),np.array(self.cmds[-1:])),axis=1)
            X = torch.tensor(X).double()
            # X[:,-1] *= 1.5
            for j in range(5) :
                X[:,:3] += model(X)*DT
            w_pred_ = float(X[-1,2].detach())
            
            X = np.concatenate((np.array(self.states[-1:]),np.array(self.cmds[-1:])),axis=1)
            X = torch.tensor(X).double()
            X[:,-1] *= -1.
            print(X)
            # X[:,1] *= -1.
            # X[:,2] *= -1.
            for j in range(20) :
                print(j,X[:,2])
                X[:,:3] += model(X)*DT
            _w_pred = float(X[-1,2].detach())

            X = np.concatenate((np.array(self.states[-10:]),np.array(self.cmds[-10:])),axis=1)
            X[:,0] = X[-1,0]
            X[:,1] = X[-1,1]
            X[:,2] = X[-1,2]
            X[:,3] = X[-1,3]
            X[:,-1] = np.arange(-1.,0.99,1./5.)
            X = torch.tensor(X).double()
            for j in range(5) :
                X[:,:3] += model(X)*DT
            print(X[:,2])
        
        odom = Odometry()
        odom.header.frame_id = 'map'
        odom.header.stamp = now
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = omega
        odom.twist.twist.angular.x = w_pred_
        odom.twist.twist.angular.y = _w_pred
        self.odom_pub_.publish(odom)
        
        # print(np.array(mppi_info['action']).shape)
        actions = np.array(mppi_info['action'])
        traj_pred = rollout_nn(env.obs_state(),actions)
        
        nn_path = Path()
        nn_path.header.frame_id = 'map'
        nn_path.header.stamp = now
        for i in range(len(traj_pred)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(traj_pred[i, 0])
            pose.pose.position.y = float(traj_pred[i, 1])
            nn_path.poses.append(pose)
        self.path_pub_nn.publish(nn_path)
        
        throttle = Float64()
        throttle.data = float(action[0])
        self.throttle_pub_.publish(throttle)
        
        steer = Float64()
        steer.data = float(action[1])
        self.steer_pub_.publish(steer)
        
        # trajectory array
        # all_trajectory is of shape horizon, num_rollout, 3
        # trajectory_array = MarkerArray()
        # for i in range(all_trajectory.shape[1]):
            # marker = Marker()
            # marker.header.frame_id = 'map'
            # marker.header.stamp = now
            # marker.type = Marker.LINE_STRIP
            # marker.action = Marker.ADD
            # marker.id = i
            # marker.scale.x = 0.05
            # marker.color.a = 1.0
            # marker.color.r = 1.0
            # marker.color.g = 0.0
            # marker.color.b = 0.0
            # for j in range(all_trajectory.shape[0]):
            #     point = all_trajectory[j, i]
            #     p = Point()
            #     p.x = float(point[0])
            #     p.y = float(point[1])
            #     p.z = 0.
            #     marker.points.append(p)
            # trajectory_array.markers.append(marker)
        # self.trajectory_array_pub_.publish(trajectory_array)
        
        # body polygon
        pts = np.array([
            [LF, L/3],
            [LF, -L/3],
            [-LR, -L/3],
            [-LR, L/3],
        ])
        # transform to world frame
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts = np.dot(R, pts.T).T
        pts += np.array([px, py])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = now
        for i in range(pts.shape[0]):
            p = Point32()
            p.x = float(pts[i, 0])
            p.y = float(pts[i, 1])
            p.z = 0.
            body.polygon.points.append(p)
        self.body_pub_.publish(body)

        
        
    def slow_timer_callback(self):
        # publish waypoint_list as path
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(waypoint_generator.waypoint_list_np.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.waypoint_list_np[i][0])
            pose.pose.position.y = float(waypoint_generator.waypoint_list_np[i][1])
            path.poses.append(pose)
        self.waypoint_list_pub_.publish(path)

def main():
    rclpy.init()
    car_node = CarNode()
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
