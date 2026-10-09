#include <librealsense2/rs.hpp>
#include <iostream>
int main() try {
 rs2::log_to_console(RS2_LOG_SEVERITY_WARN);
 rs2::pipeline p; rs2::config c;
 c.enable_device("318122302992");
 c.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,15);
 c.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_RGB8,15);
 std::cerr << "Starting 640x480 RGB+depth at 15 FPS" << std::endl;
 auto profile=p.start(c);
 std::cerr << "Pipeline started" << std::endl;
 for(int i=0;i<45;++i){
  auto f=p.wait_for_frames(5000);
  auto d=f.get_depth_frame();
  if(i%15==0) std::cout<<"frame "<<i<<" center_distance_m "<<d.get_distance(320,240)<<std::endl;
 }
 p.stop(); std::cout<<"PASS: 45 RGB-D frames"<<std::endl;
} catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1;}
