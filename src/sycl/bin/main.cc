#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include <sycl/sycl.hpp>

#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/verbose.h"
#include "EventProcessor.h"
#include "PosixClockGettime.h"

namespace {
  void print_help(std::string const& name) {
    std::cout
        << "Usage: " << name
        << " [--verbose] [--device BACKEND:DEVICE] [--numberOfThreads NT] [--numberOfStreams NS] [--warmupEvents WE]"
        << " [--maxEvents ME] [--runForMinutes RM] [--data PATH] [--transfer] [--validation] [--histogram] [--empty]\n";
    std::cout << R"(
Options:
  --device BACKEND:DEVICE       Specifies which device(s) to use (default all).
                                See https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector
                                for the accepted syntax.
  --verbose                     Write information about the devices found on the system and used by each stream.
  --numberOfThreads             Number of threads to use (default 1, use 0 to use all CPU cores).
  --numberOfStreams             Number of concurrent events (default 0 = numberOfThreads).
  --warmupEvents                Number of events to process before starting the benchmark (default 0).
  --maxEvents                   Number of events to process (default -1 for all events in the input file).
  --runForMinutes               Continue processing the set of 1000 events until this many minutes have passed
                                (default -1 for disabled; conflicts with --maxEvents).
  --data                        Path to the 'data' directory (default 'data' in the directory of the executable).
  --transfer                    Transfer results from GPU to CPU (default is to leave them on GPU).
  --validation                  Run (rudimentary) validation at the end (implies --transfer).
  --histogram                   Produce histograms at the end (implies --transfer).
  --empty                       Ignore all producers (for testing only).
)";
  }
}  // namespace

int main(int argc, char** argv) try {
  // Parse command line arguments
  std::vector<std::string> args(argv, argv + argc);
  std::string deviceSelection;
  int numberOfThreads = 1;
  int numberOfStreams = 0;
  int warmupEvents = 0;
  int maxEvents = -1;
  int runForMinutes = -1;
  std::filesystem::path datadir;
  bool transfer = false;
  bool validation = false;
  bool histogram = false;
  bool empty = false;
  for (auto i = args.begin() + 1, e = args.end(); i != e; ++i) {
    if (*i == "-h" or *i == "--help") {
      print_help(args.front());
      return EXIT_SUCCESS;
    } else if (*i == "--verbose") {
      verbose = true;
    } else if (*i == "--device") {
      ++i;
      deviceSelection = *i;
      setenv("ONEAPI_DEVICE_SELECTOR", deviceSelection.c_str(), true);
    } else if (*i == "--numberOfThreads") {
      ++i;
      numberOfThreads = std::stoi(*i);
    } else if (*i == "--numberOfStreams") {
      ++i;
      numberOfStreams = std::stoi(*i);
    } else if (*i == "--warmupEvents") {
      ++i;
      warmupEvents = std::stoi(*i);
    } else if (*i == "--maxEvents") {
      ++i;
      maxEvents = std::stoi(*i);
    } else if (*i == "--runForMinutes") {
      ++i;
      runForMinutes = std::stoi(*i);
    } else if (*i == "--data") {
      ++i;
      datadir = *i;
    } else if (*i == "--transfer") {
      transfer = true;
    } else if (*i == "--validation") {
      transfer = true;
      validation = true;
    } else if (*i == "--histogram") {
      transfer = true;
      histogram = true;
    } else if (*i == "--empty") {
      empty = true;
    } else {
      std::cout << "Invalid parameter " << *i << std::endl << std::endl;
      print_help(args.front());
      return EXIT_FAILURE;
    }
  }
  if (maxEvents >= 0 and runForMinutes >= 0) {
    std::cout << "Got both --maxEvents and --runForMinutes, please give only one of them" << std::endl;
    return EXIT_FAILURE;
  }
  if (numberOfThreads == 0) {
    numberOfThreads = tbb::info::default_concurrency();
  }
  if (numberOfStreams == 0) {
    numberOfStreams = numberOfThreads;
  }
  if (datadir.empty()) {
    datadir = std::filesystem::path(args[0]).parent_path() / "data";
  }
  if (not std::filesystem::exists(datadir)) {
    std::cout << "Data directory '" << datadir << "' does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  // Initialise the SYCL runtime
  if (cms::sycltools::enumerateDevices(verbose).empty()) {
    if (deviceSelection.empty()) {
      std::cerr << "Error: no SYCL devices found.\n";
    } else {
      std::cerr << "Error: no SYCL devices matching the selection \"" << deviceSelection << "\" found.\n";
    }
    exit(EXIT_FAILURE);
  }

  // Initialise the EventProcessor
  std::vector<std::string> edmodules;
  std::vector<std::string> esmodules;
  if (not empty) {
    edmodules = {
        "BeamSpotToSYCL", "SiPixelRawToClusterSYCL", "SiPixelRecHitSYCL", "CAHitNtupletSYCL", "PixelVertexProducerSYCL"};
    esmodules = {"BeamSpotESProducer",
                 "SiPixelFedCablingMapGPUWrapperESProducer",
                 "SiPixelGainCalibrationForHLTGPUESProducer",
                 "PixelCPEFastESProducer"};
    if (transfer) {
      auto capos = std::find(edmodules.begin(), edmodules.end(), "CAHitNtupletSYCL");
      // assert(capos != edmodules.end());
      edmodules.insert(capos + 1, "PixelTrackSoAFromSYCL");
      auto vertpos = std::find(edmodules.begin(), edmodules.end(), "PixelVertexProducerSYCL");
      // assert(vertpos != edmodules.end());
      edmodules.insert(vertpos + 1, "PixelVertexSoAFromSYCL");
    }
    if (validation) {
      edmodules.emplace_back("CountValidator");
    }
    if (histogram) {
      edmodules.emplace_back("HistoValidator");
    }
  }
  edm::EventProcessor processor(warmupEvents,
                                maxEvents,
                                runForMinutes,
                                numberOfStreams,
                                std::move(edmodules),
                                std::move(esmodules),
                                datadir,
                                validation);

  if (runForMinutes < 0) {
    std::cout << "Processing " << processor.maxEvents() << " events,";
  } else {
    std::cout << "Processing for about " << runForMinutes << " minutes,";
  }
  if (warmupEvents > 0) {
    std::cout << " after " << warmupEvents << " events of warm up,";
  }
  std::cout << " with " << numberOfStreams << " concurrent events and " << numberOfThreads << " threads." << std::endl;

  // Initialize the TBB thread pool
  tbb::global_control tbb_max_threads{tbb::global_control::max_allowed_parallelism,
                                      static_cast<std::size_t>(numberOfThreads)};

  // Warm up
  try {
    tbb::task_arena arena(numberOfThreads);
    arena.execute([&] { processor.warmUp(); });
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }

  // Run work
  auto cpu_start = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
  auto start = std::chrono::high_resolution_clock::now();
  try {
    tbb::task_arena arena(numberOfThreads);
    arena.execute([&] { processor.runToCompletion(); });
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }
  auto cpu_stop = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
  auto stop = std::chrono::high_resolution_clock::now();

  // Run endJob
  try {
    processor.endJob();
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }

  // Work done, report timing
  auto diff = stop - start;
  auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(diff).count()) / 1e6;
  auto cpu_diff = cpu_stop - cpu_start;
  auto cpu = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(cpu_diff).count()) / 1e6;
  maxEvents = processor.processedEvents();
  std::cout << "Processed " << maxEvents << " events in " << std::scientific << time << " seconds, throughput "
            << std::defaultfloat << (maxEvents / time) << " events/s, CPU usage per thread: " << std::fixed
            << std::setprecision(1) << (cpu / time / numberOfThreads * 100) << "%" << std::endl;
  return EXIT_SUCCESS;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
