#include <chrono>

class Clock {
private:
    typedef std::chrono::high_resolution_clock clock;
    std::chrono::time_point<clock> t;
public:
    Clock() { start(); }
    void start() { t = clock::now(); }
    double stop() const { return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now()-t).count(); }
};