#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    ros::init(argc, argv, "robot_arm_6dof_random_move_node");
    ros::NodeHandle nh;

    ros::AsyncSpinner spinner(1);
    spinner.start();

    static const std::string PLANNING_GROUP = "RB5";
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    const std::string end_effector_link = "end_effector";
    const std::string base_frame = "base_link";
    ROS_INFO("Using end effector link: %s", end_effector_link.c_str());
    ROS_INFO("Using base frame: %s", base_frame.c_str());
    ROS_INFO("Planning frame: %s", move_group.getPlanningFrame().c_str());

    // 초기 조인트 값으로 이동
    std::vector<double> initial_joint_values = {
        -0.280177,  // base
        -0.987132,  // shoulder
        2.338877,   // elbow
        0.242507,   // wrist1
        1.656193,   // wrist2
        -0.175954   // wrist3
    };

    move_group.setStartStateToCurrentState();
    move_group.setJointValueTarget(initial_joint_values);
    moveit::planning_interface::MoveGroupInterface::Plan initial_plan;
    bool initial_success = (move_group.plan(initial_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO("Planning to initial joint values %s", initial_success ? "SUCCEEDED" : "FAILED");

    if (initial_success) {
        ROS_INFO("Executing plan to initial joint values...");
        move_group.execute(initial_plan);
        ros::Duration(2.0).sleep();
    } else {
        ROS_ERROR("Failed to plan to initial joint values, exiting.");
        ros::shutdown();
        return 1;
    }

    // 현재 TCP 위치 가져오기 (base_link 기준)
    geometry_msgs::PoseStamped current_pose = move_group.getCurrentPose(end_effector_link);
    ROS_INFO("Current TCP position in %s: x=%f, y=%f, z=%f", 
             current_pose.header.frame_id.c_str(), 
             current_pose.pose.position.x, 
             current_pose.pose.position.y, 
             current_pose.pose.position.z);

    // 난수 생성기 설정
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.25, 0.25); // -25cm ~ +25cm

    // 기준 위치: TCP에서 z축 +50cm
    geometry_msgs::Point base_point;
    base_point.x = 0.0;
    base_point.y = 0.0;
    base_point.z = 0.5;

    // 마커 생성 및 저장
    std::vector<geometry_msgs::Pose> marker_poses(10);
    for (int i = 0; i < 10; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = end_effector_link; // TCP 기준 발행
        marker.header.stamp = ros::Time::now();
        marker.ns = "random_markers";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = base_point.x + dis(gen);
        marker.pose.position.y = base_point.y + dis(gen);
        marker.pose.position.z = base_point.z;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.03;
        marker.scale.y = 0.03;
        marker.scale.z = 0.03;

        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.7;

        marker.lifetime = ros::Duration(0);

        marker_poses[i] = marker.pose;
        marker_pub.publish(marker);

        ROS_INFO("Published marker %d (TCP frame): x=%f, y=%f, z=%f", 
                 i, marker.pose.position.x, marker.pose.position.y, marker.pose.position.z);
    }

    ros::Duration(1.0).sleep();

    // 모든 마커에 순차적으로 접근
    for (int i = 0; i < 10; ++i) {
        geometry_msgs::Pose target_pose_relative = marker_poses[i];
        ROS_INFO("Visiting marker %d (TCP frame): x=%f, y=%f, z=%f", 
                 i, target_pose_relative.position.x, target_pose_relative.position.y, target_pose_relative.position.z);

        // TCP 기준 상대 좌표를 base_link 기준 절대 좌표로 변환
        tf2::Quaternion q_current(
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        );
        tf2::Matrix3x3 rot_matrix(q_current);
        
        tf2::Vector3 relative_vec(
            target_pose_relative.position.x,
            target_pose_relative.position.y,
            target_pose_relative.position.z
        );
        tf2::Vector3 rotated_vec = rot_matrix * relative_vec;

        geometry_msgs::Pose target_pose_base;
        target_pose_base.position.x = current_pose.pose.position.x + rotated_vec.x();
        target_pose_base.position.y = current_pose.pose.position.y + rotated_vec.y();
        target_pose_base.position.z = current_pose.pose.position.z + rotated_vec.z();
        target_pose_base.orientation = current_pose.pose.orientation;
        ROS_INFO("Target pose (base_link frame): x=%f, y=%f, z=%f", 
                 target_pose_base.position.x, target_pose_base.position.y, target_pose_base.position.z);

        // 엔드 이펙터 목표 설정
        move_group.setPoseTarget(target_pose_base, end_effector_link);

        // 계획 생성
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
        ROS_INFO("Planning to marker %d with end effector %s", i, success ? "SUCCEEDED" : "FAILED");

        // 계획 실행
        if (success) {
            ROS_INFO("Executing plan to marker %d with end effector...", i);
            move_group.execute(my_plan);
            ros::Duration(2.0).sleep(); // 각 마커 방문 후 대기
        } else {
            ROS_ERROR("Failed to plan to marker %d, skipping execution.", i);
            // 실패 시 다음 마커로 진행 (실패로 인해 종료되지 않음)
        }
    }

    ROS_INFO("Visited all 10 markers.");
    ros::Duration(5.0).sleep();
    ros::shutdown();
    return 0;
}