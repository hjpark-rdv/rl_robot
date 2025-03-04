#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np

def set_random_tomato_pose():
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    # 랜덤 위치 생성 (UR3e 작업 범위 내)
    x = np.random.uniform(0.3, 0.5)
    y = np.random.uniform(-0.3, 0.3)
    z = np.random.uniform(0.2, 0.5)
    
    req = SetModelStateRequest()
    req.model_state.model_name = 'tomato'
    req.model_state.pose = Pose(
        position=Point(x, y, z),
        orientation=Quaternion(0, 0, 0, 1)
    )
    req.model_state.reference_frame = 'world'
    
    try:
        set_state(req)
        return np.array([x, y, z])
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

if __name__ == '__main__':
    rospy.init_node('random_tomato', anonymous=True)
    pos = set_random_tomato_pose()
    if pos is not None:
        rospy.loginfo(f"Set tomato position to: {pos}")
    rospy.spin()