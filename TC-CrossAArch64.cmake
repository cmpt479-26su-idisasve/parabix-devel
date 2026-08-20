# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc-15)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-15)

# where is the target environment located
# this path should look like a root path -- it should have bin, include, lib, share dirs
# those dirs should contain cross-built libraries for all the dependencies we need
set(CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../aarch64/target /usr/aarch64-linux-gnu)

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_INSTALL_PREFIX /home/jlunder/workspaces/sfu-26su-cmpt980/repos/aarch64/target)
