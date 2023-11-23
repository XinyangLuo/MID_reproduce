import numpy as np
import matplotlib.pyplot as plt

def inner_prod(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]

def outer_prod(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]

def turn_rate(x1, y1, x2, y2, x3, y3):
    v_x = (x3-x1)/2
    v_y = (y3-y1)/2
    a_x = (x1+x3-2*x2)
    a_y = (y1+y3-2*y2)
    v_norm = np.sqrt(v_x**2+v_y**2)
    k = abs(v_x*a_y - v_y*a_x) / v_norm**1.5
    r = 1/k
    return r

class BoundingBox2d():
    def __init__(self, center, vel, length, width):
        self.center = center
        self.speed = np.linalg.norm(vel)
        self.direction = vel/self.speed
        self.heading = np.arccos(self.direction)
        self.half_length = length*0.5
        self.half_width = width*0.5
        self.eps = 1e-5

    def HasOverlap(self, other_box: 'BoundingBox2d') -> bool:
        offset = other_box.center - self.center
        abs_inner = np.abs(inner_prod(self.direction, other_box.direction))
        abs_outer = np.abs(outer_prod(self.direction, other_box.direction))
        max_outer_onto_this = abs_outer * other_box.half_length + abs_inner * other_box.half_width + self.half_width
        if np.abs(outer_prod(self.direction, offset)) - max_outer_onto_this > self.eps:
            return False
        max_inner_onto_this = abs_inner * other_box.half_length + abs_outer * other_box.half_width + self.half_length
        if np.abs(inner_prod(self.direction, offset)) - max_inner_onto_this > self.eps:
            return False
        max_outer_onto_other = abs_outer * self.half_length + abs_inner * self.half_width + other_box.half_width
        if np.abs(outer_prod(offset, other_box.direction)) - max_outer_onto_other > self.eps:
            return False
        max_inner_onto_other = abs_inner * self.half_length + abs_outer * self.half_width + other_box.half_length
        if np.abs(inner_prod(offset, other_box.direction)) - max_inner_onto_other > self.eps:
            return False
        return True

    def GetCorner(self):
        d0 = self.direction * self.half_length
        d1 = np.array([-self.direction[1], self.direction[0]]) * self.half_width
        return np.array([self.center + d0 - d1,
                         self.center + d0 + d1,
                         self.center - d0 + d1,
                         self.center - d0 - d1,])

    def DistanceSquareTo(self, point):
        d = self.center - point
        dx = np.abs(inner_prod(self.direction, d)) - self.half_length
        dy = np.abs(outer_prod(self.direction, d)) - self.half_width
        return np.square(np.max([0.0, dx])) + np.square(np.max([0.0, dy]))

    def DistanceTo(self, other_box):
        if self.HasOverlap(other_box):
            return 0.0
        min_dist = float('inf')
        for p in other_box.GetCorner():
            min_dist = min(min_dist, self.DistanceSquareTo(p))
        for p in self.GetCorner():
            min_dist = min(min_dist, other_box.DistanceSquareTo(p))
        return np.sqrt(min_dist)
    

class NaturalisticMetric():
    def __init__(self, pred_traj, pred_vel, nodes):
        self.num_samples = pred_traj.shape[0]
        self.num_nodes = pred_traj.shape[1]
        self.time_length = pred_traj.shape[2]
        assert(self.num_nodes == len(nodes))
        self.pred_traj = pred_traj
        self.pred_vel = pred_vel
        self.nodes = nodes
        self.BuildAllScenes()
        print("There are {} samples, {} nodes, {} time steps".format(self.num_samples, self.num_nodes, self.time_length))

    def PosVelToBoundingBox(self, traj, vel, node_idx):
        # return t(time length) bounding boxes
        bounding_boxes = []
        for t in range(traj.shape[0]):
            center = traj[t]
            v = vel[t]
            length = self.nodes[node_idx].length
            width = self.nodes[node_idx].width
            bounding_boxes.append(BoundingBox2d(center, v, length, width))
        return bounding_boxes
    
    def BuildSceneBoundingBoxes(self, scene_idx):
        trajs = self.pred_traj[scene_idx]
        vels = self.pred_vel[scene_idx]
        node_bounding_boxes = []
        for i in range(self.num_nodes):
            node_bounding_boxes.append(self.PosVelToBoundingBox(trajs[i], vels[i], i))
        return node_bounding_boxes
    
    def BuildAllScenes(self):
        # shape [num_samples, num_nodes, time_length] each entry is a bounding box
        self.all_scenes_bounding_boxes = []
        for scene_idx in range(self.num_samples):
            self.all_scenes_bounding_boxes.append(self.BuildSceneBoundingBoxes(scene_idx))

    def DistanceDistribution(self):
        distances = []
        for i in range(self.num_samples):
            for t in range(self.time_length):
                for j in range(self.num_nodes):
                    for k in range(j+1, self.num_nodes):
                        bound_box_1 = self.all_scenes_bounding_boxes[i][j][t]
                        bound_box_2 = self.all_scenes_bounding_boxes[i][k][t]
                        dist = bound_box_1.DistanceTo(bound_box_2)
                        distances.append(dist)
        plt.hist(distances, bins=50)
        plt.grid()
        plt.title("Distance Distribution")
        plt.show()

    def SpeedDistribution(self):
        speeds = np.linalg.norm(self.pred_vel, axis=-1)
        speeds = speeds.flatten()
        plt.hist(speeds, bins=50)
        plt.grid()
        plt.title("Speed Distribution")
        plt.show()

    def CollisionDistribution(self):
        collision_angles = []
        collision_speeds = []
        num_collisions = 0 
        for i in range(self.num_samples):
            for j in range(self.num_nodes):
                for k in range(j+1, self.num_nodes):
                    for t in range(self.time_length):
                        bound_box_1 = self.all_scenes_bounding_boxes[i][j][t]
                        bound_box_2 = self.all_scenes_bounding_boxes[i][k][t]
                        if bound_box_1.HasOverlap(bound_box_2):
                            collision_speeds.append(bound_box_1.speed)
                            collision_speeds.append(bound_box_2.speed)
                            angle = np.arccos(np.dot(bound_box_1.direction, bound_box_2.direction))
                            collision_angles.append(angle)
                            num_collisions += 1
                            break

        print("num of collisions in {} samples: {}".format(self.num_samples, num_collisions))
        plt.hist(collision_speeds, bins=25)
        plt.grid()
        plt.title("Collision Speed Distribution")
        plt.show()

        plt.hist(collision_angles, bins=25)
        plt.grid()
        plt.title("Collision Angle Distribution")
        plt.show()

    def turn_rate_violation_distribution(self, cutoff=4):
        x1 = self.pred_traj[..., :self.time_length-2, 0].flatten()
        y1 = self.pred_traj[..., :self.time_length-2, 1].flatten()
        x2 = self.pred_traj[..., 1:self.time_length-1, 0].flatten()
        y2 = self.pred_traj[..., 1:self.time_length-1, 1].flatten()
        x3 = self.pred_traj[..., 2:self.time_length, 0].flatten()
        y3 = self.pred_traj[..., 2:self.time_length, 1].flatten()
        TR = turn_rate(x1, y1, x2, y2, x3, y3)
        TRV = [max(cutoff-tr, 0) for tr in TR]
        print("ATRV: {}".format(np.mean(TRV)))
        plt.hist(TRV, bins=100)
        plt.grid()
        plt.title("TRV distribution")
        plt.show()
