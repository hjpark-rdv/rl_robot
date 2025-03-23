#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <random>

int main(int argc, char** argv) {
    // ROS 노드 초기화
    ros::init(argc, argv, "random_marker_node");
    ros::NodeHandle nh;

    // 비동기 스피너 설정
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // MoveIt! MoveGroup 인터페이스 설정
    static const std::string PLANNING_GROUP = "RB5";
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);

    // 마커 퍼블리셔 설정
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    // TCP 프레임 명시
    std::string tcp_frame = move_group.getEndEffectorLink();
    if (tcp_frame.empty()) {
        tcp_frame = "link6"; // 기본값, MoveIt! 설정에 따라 변경
        move_group.setEndEffectorLink(tcp_frame);
    }
    ROS_INFO("Using TCP frame: %s", tcp_frame.c_str());

    // 난수 생성기 설정 (-25cm ~ +25cm)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.25, 0.25);

    // 현재 TCP 위치 가져오기
    geometry_msgs::PoseStamped current_pose = move_group.getCurrentPose();
    ROS_INFO("TCP Position in %s: x=%f, y=%f, z=%f", 
             current_pose.header.frame_id.c_str(),
             current_pose.pose.position.x,
             current_pose.pose.position.y,
             current_pose.pose.position.z);

    // 기준 위치: TCP에서 z축 방향으로 +50cm 이동
    geometry_msgs::Point base_point;
    base_point.x = 0.0;  // TCP 프레임 기준 로컬 좌표
    base_point.y = 0.0;
    base_point.z = 0.5;  // +50cm

    // 10개의 마커 생성 및 발행
    for (int i = 0; i < 10; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = tcp_frame; // TCP 프레임으로 고정
        marker.header.stamp = ros::Time::now();
        marker.ns = "random_markers";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;

        // 마커 위치: TCP 프레임 기준 로컬 좌표로 설정
        marker.pose.position.x = base_point.x + dis(gen); // -25cm ~ +25cm
        marker.pose.position.y = base_point.y + dis(gen); // -25cm ~ +25cm
        marker.pose.position.z = base_point.z;           // +50cm 고정
        marker.pose.orientation.w = 1.0; // 기본 방향

        // 마커 크기 설정
        marker.scale.x = 0.03;
        marker.scale.y = 0.03;
        marker.scale.z = 0.03;

        // 마커 색상 설정 (초록색, 반투명)
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.7;

        // 마커 지속 시간 설정 (영구)
        marker.lifetime = ros::Duration(0);

        // 마커 발행
        marker_pub.publish(marker);
    }

    ROS_INFO("Published 10 random markers at z+50cm from TCP in frame %s", tcp_frame.c_str());

    // 발행 후 잠시 대기 (RViz가 마커를 받을 시간 확보)
    ros::Duration(1.0).sleep();

    // ROS 종료
    ros::shutdown();
    return 0;
}