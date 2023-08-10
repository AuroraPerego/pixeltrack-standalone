#ifndef bin_Timestamp_h
#define bin_Timestamp_h

#include <chrono>

#include "PosixClockGettime.h"

struct Timestamp {
  PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::time_point cpu;
  std::chrono::high_resolution_clock::time_point real;

  void mark() {
    cpu = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
    real = std::chrono::high_resolution_clock::now();
  }

  static Timestamp now() {
    Timestamp t;
    t.mark();
    return t;
  }
};

#endif  // bin_Timestamp_h
