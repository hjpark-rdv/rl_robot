cmake_minimum_required(VERSION 3.0.2)
project(robot_arm_6dof)

# Catkin 의존성 설정
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  visualization_msgs  
)

# Catkin 패키지 설정
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs
)

# 인클루드 디렉토리 추가
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# 실행 파일 생성
add_executable(robot_arm_6dof src/robot_arm_6dof.cpp)
add_executable(random_marker src/random_marker.cpp)

target_link_libraries(robot_arm_6dof ${catkin_LIBRARIES} )
target_link_libraries(random_marker ${catkin_LIBRARIES} )

# 설치 설정 (필요 시)
install(TARGETS robot_arm_6dof RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS random_marker RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})