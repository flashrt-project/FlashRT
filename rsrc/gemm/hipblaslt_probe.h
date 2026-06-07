#pragma once

#include <string>

struct HipblasLtProbeResult {
  bool available = false;
  int status_code = -1;
  std::string status_name;
  int version = 0;
};

HipblasLtProbeResult probe_hipblaslt();
