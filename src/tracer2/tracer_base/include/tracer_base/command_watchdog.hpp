#pragma once
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace westonrobot {
class CommandWatchdog {
 public:
  using Clock = std::chrono::steady_clock;
  explicit CommandWatchdog(double seconds = 0.0) : seconds_(seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0)
      throw std::invalid_argument("command_timeout must be finite and nonnegative");
  }
  void Received(Clock::time_point now = Clock::now()) { last_ = now; received_ = true; }
  bool Expired(Clock::time_point now = Clock::now()) const {
    return seconds_ > 0.0 && (!received_ ||
      std::chrono::duration<double>(now - last_).count() >= seconds_);
  }
 private:
  double seconds_;
  bool received_ = false;
  Clock::time_point last_{};
};
}  // namespace westonrobot
