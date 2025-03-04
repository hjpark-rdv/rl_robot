#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def set_initial_pose():
    rospy.init_node('set_initial_pose', anonymous=True)
    pub = rospy.Publisher('/ur3e/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)
    rospy.sleep(1)  # 컨트롤러 초기화 대기

    msg = JointTrajectory()
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()
    point.positions = [0.0, -2.57, 1.57, 0.0, 1.57, 0.0]  # 똑바로 서는 자세
    point.time_from_start = rospy.Duration(2.0)
    msg.points.append(point)

    pub.publish(msg)
    rospy.loginfo("Published initial pose")
    rospy.sleep(2)  # 메시지 전송 대기

if __name__ == '__main__':
    try:
        set_initial_pose()
    except rospy.ROSInterruptException:
        pass