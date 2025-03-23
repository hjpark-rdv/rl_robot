#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <random>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    ros::init(argc, argv, "robot_arm_6dof_random_move_node");
    ros::NodeHandle nh;

    ros::AsyncSpinner spinner(1);
    spinner.start();

    static const std::string PLANNING_GROUP = "RB5";
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    const std::string end_effector_link = "end_effector";
    const std::string base_frame = "Body_Base";
    ROS_INFO("Using end effector link: %s", end_effector_link.c_str());
    ROS_INFO("Using base frame: %s", base_frame.c_str());
    ROS_INFO("Planning frame: %s", move_group.getPlanningFrame().c_str());

    // 초기 조인트 값 정의
    std::vector<double> initial_joint_values = {
        -0.280177,  // base
        -0.987132,  // shoulder
        2.338877,   // elbow
        0.242507,   // wrist1
        1.656193,   // wrist2
        -0.175954   // wrist3
    };

    // 처음 시작 시 초기 조인트로 이동
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
    ROS_INFO("Current TCP orientation: w=%f, x=%f, y=%f, z=%f", 
             current_pose.pose.orientation.w, 
             current_pose.pose.orientation.x, 
             current_pose.pose.orientation.y, 
             current_pose.pose.orientation.z);

    // 난수 생성기 설정
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.25, 0.25); // -25cm ~ +25cm

    // 기준 위치: TCP에서 z축 +50cm
    geometry_msgs::Point base_point;
    base_point.x = 0.0;
    base_point.y = 0.0;
    base_point.z = 0.5;

    // 단일 마커 (random object1) 생성
    visualization_msgs::Marker marker;
    marker.header.frame_id = end_effector_link; // TCP 기준 발행
    marker.header.stamp = ros::Time::now();
    marker.ns = "random_objects";
    marker.id = 1; // object1
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

    marker_pub.publish(marker);
    ROS_INFO("Published random object1 (TCP frame): x=%f, y=%f, z=%f", 
             marker.pose.position.x, marker.pose.position.y, marker.pose.position.z);

    ros::Duration(1.0).sleep();

    // 단일 마커에 접근
    geometry_msgs::Pose target_pose_relative = marker.pose;
    ROS_INFO("Visiting random object1 (TCP frame): x=%f, y=%f, z=%f", 
             target_pose_relative.position.x, target_pose_relative.position.y, target_pose_relative.position.z);

    // TCP 기준 상대 좌표를 base_link 기준 절대 좌표로 변환
    geometry_msgs::PoseStamped target_pose_stamped;
    target_pose_stamped.header.frame_id = end_effector_link;
    target_pose_stamped.header.stamp = ros::Time::now();
    target_pose_stamped.pose = target_pose_relative;

    geometry_msgs::TransformStamped transform;
    transform.header.stamp = ros::Time::now();
    transform.header.frame_id = base_frame;
    transform.child_frame_id = end_effector_link;
    transform.transform.translation.x = current_pose.pose.position.x;
    transform.transform.translation.y = current_pose.pose.position.y;
    transform.transform.translation.z = current_pose.pose.position.z;
    transform.transform.rotation = current_pose.pose.orientation;

    geometry_msgs::PoseStamped target_pose_base_stamped;
    try {
        tf2::doTransform(target_pose_stamped, target_pose_base_stamped, transform);
        ROS_INFO("Target pose (base_link frame): x=%f, y=%f, z=%f", 
                 target_pose_base_stamped.pose.position.x, 
                 target_pose_base_stamped.pose.position.y, 
                 target_pose_base_stamped.pose.position.z);
    } catch (tf2::TransformException &ex) {
        ROS_ERROR("Failed to transform pose: %s", ex.what());
        ros::shutdown();
        return 1;
    }

    // 현재 방향에서 z축 벡터 계산
    tf2::Quaternion q_current(
        current_pose.pose.orientation.x,
        current_pose.pose.orientation.y,
        current_pose.pose.orientation.z,
        current_pose.pose.orientation.w
    );
    tf2::Matrix3x3 rot_matrix(q_current);
    tf2::Vector3 z_axis(0.0, 0.0, 1.0); // 로컬 z축
    tf2::Vector3 z_axis_global = rot_matrix * z_axis; // base_link 기준 z축

    // 목표 방향 벡터 계산
    tf2::Vector3 current_pos(
        current_pose.pose.position.x,
        current_pose.pose.position.y,
        current_pose.pose.position.z
    );
    tf2::Vector3 target_pos(
        target_pose_base_stamped.pose.position.x,
        target_pose_base_stamped.pose.position.y,
        target_pose_base_stamped.pose.position.z
    );
    tf2::Vector3 target_vec = (target_pos - current_pos).normalized();

    // Roll 각도 계산 (x축 기준, 2배 회전 보정)
    double roll_angle = atan2(z_axis_global.y(), z_axis_global.z()) - atan2(target_vec.y(), target_vec.z());
    roll_angle /= 2.0; // 회전량 절반으로 조정

    ROS_INFO("Calculated roll_angle (radians): %f", roll_angle);

    // Roll만 적용 (pitch와 yaw는 0)
    tf2::Quaternion q_roll;
    q_roll.setRPY(roll_angle, 0.0, 0.0); // roll만 적용
    q_roll.normalize();
    tf2::Quaternion q_new = q_current * q_roll;

    // 회전 적용
    geometry_msgs::Pose rotate_pose = current_pose.pose;
    rotate_pose.orientation.x = q_new.x();
    rotate_pose.orientation.y = q_new.y();
    rotate_pose.orientation.z = q_new.z();
    rotate_pose.orientation.w = q_new.w();

    // 2. 목표 방향으로 실제 회전 (Roll만)
    move_group.setPoseTarget(rotate_pose, end_effector_link);
    moveit::planning_interface::MoveGroupInterface::Plan rotate_plan;
    bool rotate_success = (move_group.plan(rotate_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO("Planning to rotate (roll only) to random object1 %s", rotate_success ? "SUCCEEDED" : "FAILED");

    if (rotate_success) {
        ROS_INFO("Executing rotation (roll only) to random object1...");
        move_group.execute(rotate_plan);
        ros::Duration(1.0).sleep();
    } else {
        ROS_ERROR("Failed to execute rotation to random object1, exiting.");
        ros::shutdown();
        return 1;
    }

    // 회전 후 현재 방향 가져오기
    geometry_msgs::PoseStamped rotated_pose = move_group.getCurrentPose(end_effector_link);
    ROS_INFO("After rotation, current TCP orientation: w=%f, x=%f, y=%f, z=%f", 
             rotated_pose.pose.orientation.w, 
             rotated_pose.pose.orientation.x, 
             rotated_pose.pose.orientation.y, 
             rotated_pose.pose.orientation.z);

    // 3. 목표 위치로 실제 직선 이동 (회전된 방향 고정)
    geometry_msgs::Pose target_pose_base = target_pose_base_stamped.pose;
    target_pose_base.orientation = rotated_pose.pose.orientation; // 회전 후 방향 사용

    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(rotated_pose.pose); // 회전 후 시작 위치
    waypoints.push_back(target_pose_base);  // 목표 위치

    moveit_msgs::RobotTrajectory trajectory;
    const double eef_step = 0.01;
    const double jump_threshold = 0.0;
    double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

    if (fraction > 0.9) {
        moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
        cartesian_plan.trajectory_ = trajectory;
        ROS_INFO("Planning to move to random object1 with Cartesian path succeeded (fraction: %f)", fraction);
        ROS_INFO("Executing Cartesian move to random object1 with end effector...");
        move_group.execute(cartesian_plan);
        ros::Duration(2.0).sleep();
    } else {
        ROS_ERROR("Failed to execute Cartesian move to random object1 (fraction: %f), exiting.", fraction);
        ros::shutdown();
        return 1;
    }

    ROS_INFO("Visited random object1.");
    ros::Duration(5.0).sleep();
    ros::shutdown();
    return 0;
}