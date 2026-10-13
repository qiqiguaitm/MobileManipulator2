#include <gtest/gtest.h>
#include <explore/frontier_search.h>
#include <nav2_costmap_2d/cost_values.hpp>
using nav2_costmap_2d::Costmap2D;
using frontier_exploration::FrontierSearch;

TEST(FrontierSearch, InflationBridgeReachesForwardFrontiersAndTargetsKnownCells) {
  Costmap2D map(60, 30, .1, -3., -1.5, 255);
  for (unsigned y=7; y<23; ++y)
    for (unsigned x=5; x<55; ++x) map.setCost(x,y,(x>=30 && x<=35) ? 80 : 0);
  FrontierSearch search(&map,3.,1.,.3,rclcpp::get_logger("test"));
  geometry_msgs::msg::Point robot;robot.x=-.5;
  auto frontiers=search.searchFrom(robot);ASSERT_FALSE(frontiers.empty());
  bool front=false;
  for (const auto& f:frontiers) {
    for (const auto& p:f.points) if(p.x>2.)front=true;
    unsigned x,y;ASSERT_TRUE(map.worldToMap(f.travel_point.x,f.travel_point.y,x,y));
    EXPECT_LT(map.getCost(x,y),253);
    double cx=0.,cy=0.;for(const auto& p:f.points){cx+=p.x;cy+=p.y;}
    EXPECT_EQ(f.size,f.points.size());
    EXPECT_NEAR(f.centroid.x,cx/f.size,1e-9);EXPECT_NEAR(f.centroid.y,cy/f.size,1e-9);
  }
  EXPECT_TRUE(front);
}
TEST(FrontierSearch, DoesNotCrossInscribedOrLethalBarrier) {
  for (unsigned char blocked : {253,254}) {
    Costmap2D map(60,30,.1,-3.,-1.5,255);
    for(unsigned y=7;y<23;++y)
      for(unsigned x=5;x<55;++x)map.setCost(x,y,x>=30&&x<=35?blocked:0);
    FrontierSearch search(&map,3.,1.,.3,rclcpp::get_logger("test"));
    geometry_msgs::msg::Point robot;robot.x=-.5;
    auto frontiers=search.searchFrom(robot);ASSERT_FALSE(frontiers.empty());
    for(const auto& f:frontiers)EXPECT_LT(f.travel_point.x,0.);
  }
}
TEST(FrontierSearch, BlockedRobotCellDoesNotJumpToRemoteFreeIsland) {
  Costmap2D map(20,20,.1,-1.,-1.,255);map.setCost(5,5,0);
  FrontierSearch search(&map,3.,1.,.1,rclcpp::get_logger("test"));
  geometry_msgs::msg::Point robot;EXPECT_TRUE(search.searchFrom(robot).empty());
}
