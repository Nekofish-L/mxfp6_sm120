#pragma once
#include "mxfp_common/workspace.hpp"
namespace mxfp8_runtime {
struct WorkspaceTag {};
using WorkspacePool = mxfp_common::WorkspacePool<WorkspaceTag>;
WorkspacePool& workspace_pool();
}
