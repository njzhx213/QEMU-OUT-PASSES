cmake_minimum_required(VERSION 3.20)
project(qemu-passes LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find LLVM / MLIR / CIRCT ---
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

# Include dirs
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${CIRCT_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

# --- TableGen: generate Passes.h.inc ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes.td)

set(_TBLGEN_I_ARGS)
foreach(inc_dir ${CIRCT_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  list(APPEND _TBLGEN_I_ARGS -I${inc_dir})
endforeach()

mlir_tablegen(Passes.h.inc -gen-pass-decls ${_TBLGEN_I_ARGS})
add_public_tablegen_target(QemuPassesIncGen)

# --- Build the executable ---
add_executable(qemu-passes
  tool_main.cpp
  lib/Passes.cpp
)
add_dependencies(qemu-passes QemuPassesIncGen)

target_link_libraries(qemu-passes PRIVATE
  # MLIR core
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRAnalysis
  MLIRRewrite
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRControlFlowDialect

  # CIRCT dialects
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSeq
  CIRCTLLHD

  # LLVM support
  LLVMSupport
)

target_compile_options(qemu-passes PRIVATE -fno-rtti)
