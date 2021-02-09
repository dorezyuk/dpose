import rospy
import numpy as np
from geometry_msgs.msg import PolygonStamped, PoseStamped, Point32
from tf.transformations import euler_from_quaternion


def get_yaw(pose: PoseStamped):
    """Get the yaw from PoseStamped"""
    q = pose.pose.orientation
    return euler_from_quaternion((q.x, q.y, q.z, q.w))[2]


def get_rot_matrix(pose: PoseStamped):
    """Get the rotation matrix from PoseStamped"""
    yaw = get_yaw(pose)
    return np.matrix([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


def get_trans_vector(pose: PoseStamped):
    """Get the translation from PoseStamped"""
    return np.matrix([[pose.pose.position.x, pose.pose.position.y]]).T


class FootprintTransformer(object):

    def __init__(self):
        self.footprint = rospy.get_param("/footprint")
        self.sub = rospy.Subscriber(
            "/navigation/move_base_flex/dpose_goal_tolerance/filtered", PoseStamped, self.pose_callback, queue_size=1)
        self.pub = rospy.Publisher("filtered", PolygonStamped, queue_size=1)

    def pose_callback(self, msg: PoseStamped):
        # transform the footprint into the pose
        original = np.matrix(self.footprint).T
        rotated = get_rot_matrix(msg) * original + get_trans_vector(msg)
        # transform so we can iterate simpler
        rotated = rotated.T

        # generate the polygon
        out = PolygonStamped()
        out.header.frame_id = msg.header.frame_id
        out.polygon.points = [
            Point32(x=col.item(0), y=col.item(1)) for col in rotated]
        
        # publish the result
        self.pub.publish(out)


if __name__ == '__main__':
    rospy.init_node("footprint_transformer")
    _ = FootprintTransformer()
    rospy.spin()
