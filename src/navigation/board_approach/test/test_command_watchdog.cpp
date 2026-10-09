#include <cassert>
#include <chrono>
#include <limits>
#include "tracer_base/command_watchdog.hpp"
int main() {
  using westonrobot::CommandWatchdog;
  using Clock = CommandWatchdog::Clock;
  const auto t = Clock::time_point{} + std::chrono::seconds(10);
  CommandWatchdog guard(.4);
  assert(guard.Expired(t)); // No command received: stationary.
  guard.Received(t);
  assert(!guard.Expired(t + std::chrono::milliseconds(399)));
  assert(guard.Expired(t + std::chrono::milliseconds(400)));
  guard.Received(t + std::chrono::milliseconds(500));
  assert(!guard.Expired(t + std::chrono::milliseconds(700)));
  assert(guard.Expired(t + std::chrono::seconds(1)));
  CommandWatchdog legacy;
  assert(!legacy.Expired(t)); // Default preserves other existing launch behaviour.
  bool invalid = false;
  try { CommandWatchdog bad(std::numeric_limits<double>::quiet_NaN()); }
  catch (const std::invalid_argument&) { invalid = true; }
  assert(invalid);
}
