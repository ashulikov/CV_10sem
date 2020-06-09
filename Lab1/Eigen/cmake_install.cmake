# Install script for directory: C:/eigen-3.3.7/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/Eigen3")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "C:/eigen-3.3.7/Eigen/Cholesky"
    "C:/eigen-3.3.7/Eigen/CholmodSupport"
    "C:/eigen-3.3.7/Eigen/Core"
    "C:/eigen-3.3.7/Eigen/Dense"
    "C:/eigen-3.3.7/Eigen/Eigen"
    "C:/eigen-3.3.7/Eigen/Eigenvalues"
    "C:/eigen-3.3.7/Eigen/Geometry"
    "C:/eigen-3.3.7/Eigen/Householder"
    "C:/eigen-3.3.7/Eigen/IterativeLinearSolvers"
    "C:/eigen-3.3.7/Eigen/Jacobi"
    "C:/eigen-3.3.7/Eigen/LU"
    "C:/eigen-3.3.7/Eigen/MetisSupport"
    "C:/eigen-3.3.7/Eigen/OrderingMethods"
    "C:/eigen-3.3.7/Eigen/PaStiXSupport"
    "C:/eigen-3.3.7/Eigen/PardisoSupport"
    "C:/eigen-3.3.7/Eigen/QR"
    "C:/eigen-3.3.7/Eigen/QtAlignedMalloc"
    "C:/eigen-3.3.7/Eigen/SPQRSupport"
    "C:/eigen-3.3.7/Eigen/SVD"
    "C:/eigen-3.3.7/Eigen/Sparse"
    "C:/eigen-3.3.7/Eigen/SparseCholesky"
    "C:/eigen-3.3.7/Eigen/SparseCore"
    "C:/eigen-3.3.7/Eigen/SparseLU"
    "C:/eigen-3.3.7/Eigen/SparseQR"
    "C:/eigen-3.3.7/Eigen/StdDeque"
    "C:/eigen-3.3.7/Eigen/StdList"
    "C:/eigen-3.3.7/Eigen/StdVector"
    "C:/eigen-3.3.7/Eigen/SuperLUSupport"
    "C:/eigen-3.3.7/Eigen/UmfPackSupport"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "C:/eigen-3.3.7/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

