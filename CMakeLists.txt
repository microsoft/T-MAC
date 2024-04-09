cmake_minimum_required(VERSION 3.18)
project(TMAC C CXX)

set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TVM_HOME)
  if(DEFINED ENV{TVM_HOME})
    set(TVM_HOME "$ENV{TVM_HOME}")
  else()
    set(TVM_HOME 3rdparty/tvm)
  endif(DEFINED ENV{TVM_HOME})
endif (NOT DEFINED TVM_HOME)
message(STATUS "TVM_HOME: ${TVM_HOME}")
add_subdirectory(${TVM_HOME} tvm EXCLUDE_FROM_ALL)

set(
  TMAC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${TVM_HOME}/include/
  ${TVM_HOME}/3rdparty/dmlc-core/include/
  ${TVM_HOME}/3rdparty/dlpack/include/
)

set(KERNEL_DEPLOY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deploy)

# add_custom_command(
#   OUTPUT ${KERNEL_DEPLOY_DIR}/tuned/kernels.o kernels.o
#   COMMAND python ${KERNEL_DEPLOY_DIR}/compile.py -t -r -o ${KERNEL_DEPLOY_DIR}/tuned -da -d intel_win -b 4 -nt 1 -tb
#   VERBATIM
# )

add_library(t_mac_kernels_objs OBJECT IMPORTED)
set_target_properties(
  t_mac_kernels_objs
  PROPERTIES
    IMPORTED_OBJECTS ${KERNEL_DEPLOY_DIR}/tuned/kernels.o
)

add_library(t_mac_kernels STATIC $<TARGET_OBJECTS:t_mac_kernels_objs>)
set_target_properties(t_mac_kernels PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(t_mac_kernels tvm_runtime)

set(
  TMAC_COMPILE_DEFS
  DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
)
set(
  TMAC_LIB_INSTALL_DIR
  ${CMAKE_INSTALL_LIBDIR}
  CACHE PATH "Location of library files"
)
set(
  TMAC_INCLUDE_INSTALL_DIR
  ${CMAKE_INSTALL_INCLUDEDIR}
  CACHE PATH "Location of header files"
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TMACConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TMACConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/t-mac
  PATH_VARS
    TMAC_LIB_INSTALL_DIR
    TMAC_INCLUDE_INSTALL_DIR
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TMACConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/t-mac
)

install(TARGETS t_mac_kernels tvm_runtime LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${TMAC_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  FILES ${KERNEL_DEPLOY_DIR}/tuned/kcfg.ini
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
