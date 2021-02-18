import rospy
import rosbag
import argparse
from geometry_msgs.msg import PoseStamped

if __name__ == '__main__':
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="path to the bag file containing the poses", type=str)
    args = parser.parse_args()

    rospy.init_node('pose_publisher')

    # configure the ros-comm
    rate = rospy.Rate(1)
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    bag = rosbag.Bag(args.bagfile)

    # publish the messages
    for topic, msg, t in bag.read_messages(topics=['/move_base_simple/goal']):
        pub.publish(msg)
        rate.sleep()
        