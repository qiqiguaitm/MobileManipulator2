#include <librealsense2/rs.hpp>
#include <iostream>
int main(){rs2::context c;auto d=c.query_devices();for(auto x:d){if(std::string(x.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))=="318122302992"){x.hardware_reset();std::cout<<"D455 hardware reset sent"<<std::endl;return 0;}}return 1;}
