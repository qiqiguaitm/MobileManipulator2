/*
 * HiPNUC IMU ROS2 Driver
 * Modified for MobileManipulator2 project
 *
 * Changes from original:
 * - Fixed while(1) to while(rclcpp::ok()) for proper shutdown
 * - Added serial port open failure check
 * - Added destructor for proper resource cleanup
 * - Improved error handling in imu_read()
 * - Use system time directly for timestamp (same as Lidar driver)
 *   This ensures IMU and Lidar timestamps are aligned for FAST-LIO
 */

#include <iostream>
#include <cstring>
#include <chrono>
#include <sensor_msgs/msg/imu.hpp>
#include "rclcpp/rclcpp.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <asm/termbits.h>
#include <sys/ioctl.h>

#ifdef __cplusplus
extern "C"{
#endif
#include <poll.h>

#include "hipnuc_lib_package/hipnuc_dec.h"

#define GRA_ACC     (9.8)
#define DEG_TO_RAD  (0.01745329)
#define BUF_SIZE    (1024)
#ifdef __cplusplus
}
#endif



namespace hipnuc_driver
{
	using namespace std::chrono_literals;
	using namespace std;
	static hipnuc_raw_t raw;

	class IMUPublisher : public rclcpp::Node
	{
		public:
			int fd = -1;  // Initialize to invalid value
			uint8_t buf[BUF_SIZE] = {0};

			IMUPublisher(const rclcpp::NodeOptions &options) : Node("IMU_publisher", options)
			{
				this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
				this->declare_parameter<int>("baud_rate", 115200);
				this->declare_parameter<std::string>("frame_id", "base_link");
				this->declare_parameter<std::string>("imu_topic", "/IMU_data");

				this->get_parameter("serial_port", serial_port);
				this->get_parameter("baud_rate", baud_rate);
				this->get_parameter("frame_id", frame_id);
				this->get_parameter("imu_topic", imu_topic);

				RCLCPP_INFO(this->get_logger(), "serial_port: %s", serial_port.c_str());
				RCLCPP_INFO(this->get_logger(), "baud_rate: %d", baud_rate);
				RCLCPP_INFO(this->get_logger(), "frame_id: %s", frame_id.c_str());
				RCLCPP_INFO(this->get_logger(), "imu_topic: %s", imu_topic.c_str());

				imu_pub = this->create_publisher<sensor_msgs::msg::Imu>(imu_topic, 20);

				// Open serial port
				fd = open_ttyport(serial_port, baud_rate);

				// Check if serial port opened successfully
				if (fd < 0) {
					RCLCPP_FATAL(this->get_logger(), "Failed to open serial port: %s", serial_port.c_str());
					rclcpp::shutdown();
					return;
				}

				RCLCPP_INFO(this->get_logger(), "IMU driver started successfully");

				// Main loop - use rclcpp::ok() instead of while(1) for proper shutdown
				while(rclcpp::ok()) {
					imu_read();
				}

				// Clean up serial port
				if (fd >= 0) {
					close(fd);
					fd = -1;
					RCLCPP_INFO(this->get_logger(), "Serial port closed");
				}
			}

			// Destructor for proper cleanup
			~IMUPublisher() {
				if (fd >= 0) {
					close(fd);
					RCLCPP_INFO(this->get_logger(), "Serial port closed in destructor");
				}
			}

		private:
			// For debug logging
			bool first_msg_logged_ = false;

			// Get current system time in seconds (same method as rslidar)
			double get_system_time_sec() {
				auto now = std::chrono::system_clock::now();
				auto duration = now.time_since_epoch();
				auto sec = std::chrono::duration_cast<std::chrono::seconds>(duration);
				auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(duration - sec);
				return sec.count() + nsec.count() / 1e9;
			}

			// Convert timestamp (in seconds) to ROS message stamp
			void set_timestamp(sensor_msgs::msg::Imu& msg, double timestamp_sec) {
				int32_t sec = static_cast<int32_t>(timestamp_sec);
				uint32_t nsec = static_cast<uint32_t>((timestamp_sec - sec) * 1e9);
				msg.header.stamp.sec = sec;
				msg.header.stamp.nanosec = nsec;
			}

			void imu_read(void)
			{
				struct pollfd p;
				p.fd = fd;
				p.events = POLLIN;

				int rpoll = poll(&p, 1, 1);  // 1ms timeout

				// Return on timeout or error
				if(rpoll <= 0)
					return;

				int n = read(fd, buf, sizeof(buf));

				// Check read result
				if (n <= 0) {
					if (n < 0 && errno != EAGAIN) {
						RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
							"Serial read error: %s", strerror(errno));
					}
					return;
				}

				for(int i = 0; i < n; i++)
				{
					int rev = hipnuc_input(&raw, buf[i]);

                    if(rev)
                    {
                        auto imu_msg = std::make_unique<sensor_msgs::msg::Imu>();

                        // Extract IMU hardware timestamp
                        uint64_t imu_time_us = 0;
                        bool has_hw_timestamp = false;

                        if (raw.hi83.tag == 0x83)
                        {
                            uint32_t bm = raw.hi83.data_bitmap;

                            // Extract IMU data
                            if (bm & HI83_BMAP_QUAT)
                            {
                                imu_msg->orientation.w = raw.hi83.quat[0];
                                imu_msg->orientation.x = raw.hi83.quat[1];
                                imu_msg->orientation.y = raw.hi83.quat[2];
                                imu_msg->orientation.z = raw.hi83.quat[3];
                            }
                            if (bm & HI83_BMAP_GYR_B)
                            {
                                imu_msg->angular_velocity.x = raw.hi83.gyr_b[0];
                                imu_msg->angular_velocity.y = raw.hi83.gyr_b[1];
                                imu_msg->angular_velocity.z = raw.hi83.gyr_b[2];
                            }
                            if (bm & HI83_BMAP_ACC_B)
                            {
                                imu_msg->linear_acceleration.x = raw.hi83.acc_b[0];
                                imu_msg->linear_acceleration.y = raw.hi83.acc_b[1];
                                imu_msg->linear_acceleration.z = raw.hi83.acc_b[2];
                            }

                            // Get hardware timestamp (microseconds)
                            if (bm & HI83_BMAP_SYSTEM_TIME)
                            {
                                imu_time_us = raw.hi83.system_time_us;
                                has_hw_timestamp = true;
                            }
                        }
                        else if (raw.hi91.tag == 0x91)
                        {
                            imu_msg->orientation.w = raw.hi91.quat[0];
                            imu_msg->orientation.x = raw.hi91.quat[1];
                            imu_msg->orientation.y = raw.hi91.quat[2];
                            imu_msg->orientation.z = raw.hi91.quat[3];
                            imu_msg->angular_velocity.x = raw.hi91.gyr[0] * DEG_TO_RAD;
                            imu_msg->angular_velocity.y = raw.hi91.gyr[1] * DEG_TO_RAD;
                            imu_msg->angular_velocity.z = raw.hi91.gyr[2] * DEG_TO_RAD;
                            imu_msg->linear_acceleration.x = raw.hi91.acc[0] * GRA_ACC;
                            imu_msg->linear_acceleration.y = raw.hi91.acc[1] * GRA_ACC;
                            imu_msg->linear_acceleration.z = raw.hi91.acc[2] * GRA_ACC;

                            // Get hardware timestamp (milliseconds -> microseconds)
                            imu_time_us = static_cast<uint64_t>(raw.hi91.system_time) * 1000;
                            has_hw_timestamp = true;
                        }

                        imu_msg->header.frame_id = frame_id;

                        // === TIMESTAMP STRATEGY ===
                        // Use system time directly (same as Lidar driver with use_lidar_clock=0)
                        // This ensures IMU and Lidar timestamps are aligned for FAST-LIO
                        double system_time_sec = get_system_time_sec();

                        // Log first message for debugging
                        if (!first_msg_logged_ && has_hw_timestamp && imu_time_us > 0) {
                            double imu_time_sec = imu_time_us / 1e6;
                            RCLCPP_INFO(this->get_logger(),
                                "IMU using system time: hw_time=%.3fs, sys_time=%.3fs (using sys_time for FAST-LIO sync)",
                                imu_time_sec, system_time_sec);
                            first_msg_logged_ = true;
                        }

                        set_timestamp(*imu_msg, system_time_sec);
                        imu_pub->publish(std::move(imu_msg));
                    }
				}

				memset(buf, 0, sizeof(buf));
			}

			int open_ttyport(std::string tty_port, int baud)
			{
				const char *port_device = tty_port.c_str();
				int serial_port = open(port_device, O_RDWR | O_NOCTTY | O_NONBLOCK);
				if (serial_port < 0)
				{
					RCLCPP_ERROR(this->get_logger(), "Error opening serial port %s: %s",
						port_device, strerror(errno));
					return -1;
				}

				struct termios2 tty;

				if (ioctl(serial_port, TCGETS2, &tty) != 0)
				{
					RCLCPP_ERROR(this->get_logger(), "Error from TCGETS2 ioctl: %s", strerror(errno));
					close(serial_port);
					return -1;
				}

				tty.c_cflag &= ~CBAUD;
				tty.c_cflag |= BOTHER;

				tty.c_ispeed = baud;
				tty.c_ospeed = baud;

				tty.c_cflag |= CS8;
				tty.c_cflag &= ~PARENB;
				tty.c_cflag &= ~CSTOPB;
				tty.c_cflag &= ~CRTSCTS;

				tty.c_lflag &= ~ICANON;
				tty.c_lflag &= ~ECHO;
				tty.c_lflag &= ~ECHOE;
				tty.c_lflag &= ~ECHONL;
				tty.c_lflag &= ~ISIG;

				tty.c_iflag &= ~(IXON | IXOFF | IXANY);
				tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

				tty.c_cc[VTIME] = 10;
				tty.c_cc[VMIN] = 0;

				if (ioctl(serial_port, TCSETS2, &tty) != 0)
				{
					RCLCPP_ERROR(this->get_logger(), "Error from TCSETS2 ioctl: %s", strerror(errno));
					close(serial_port);
					return -1;
				}

				return serial_port;
			}

			std::string serial_port;
			int baud_rate;
			std::string frame_id;
			std::string imu_topic;

			rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
	};
};


int main(int argc, const char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::NodeOptions options;
	options.use_intra_process_comms(true);
	rclcpp::spin(std::make_shared<hipnuc_driver::IMUPublisher>(options));
	rclcpp::shutdown();

	return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hipnuc_driver::IMUPublisher)
