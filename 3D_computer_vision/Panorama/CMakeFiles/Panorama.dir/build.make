# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.9.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.9.4/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial

# Include any dependencies generated for this target.
include CMakeFiles/Panorama.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Panorama.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Panorama.dir/flags.make

CMakeFiles/Panorama.dir/Panorama.cpp.o: CMakeFiles/Panorama.dir/flags.make
CMakeFiles/Panorama.dir/Panorama.cpp.o: Panorama.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Panorama.dir/Panorama.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Panorama.dir/Panorama.cpp.o -c /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial/Panorama.cpp

CMakeFiles/Panorama.dir/Panorama.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Panorama.dir/Panorama.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial/Panorama.cpp > CMakeFiles/Panorama.dir/Panorama.cpp.i

CMakeFiles/Panorama.dir/Panorama.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Panorama.dir/Panorama.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial/Panorama.cpp -o CMakeFiles/Panorama.dir/Panorama.cpp.s

CMakeFiles/Panorama.dir/Panorama.cpp.o.requires:

.PHONY : CMakeFiles/Panorama.dir/Panorama.cpp.o.requires

CMakeFiles/Panorama.dir/Panorama.cpp.o.provides: CMakeFiles/Panorama.dir/Panorama.cpp.o.requires
	$(MAKE) -f CMakeFiles/Panorama.dir/build.make CMakeFiles/Panorama.dir/Panorama.cpp.o.provides.build
.PHONY : CMakeFiles/Panorama.dir/Panorama.cpp.o.provides

CMakeFiles/Panorama.dir/Panorama.cpp.o.provides.build: CMakeFiles/Panorama.dir/Panorama.cpp.o


# Object files for target Panorama
Panorama_OBJECTS = \
"CMakeFiles/Panorama.dir/Panorama.cpp.o"

# External object files for target Panorama
Panorama_EXTERNAL_OBJECTS =

Panorama: CMakeFiles/Panorama.dir/Panorama.cpp.o
Panorama: CMakeFiles/Panorama.dir/build.make
Panorama: /Users/raphaelromero/anaconda/lib/libQt5OpenGL.5.6.0.dylib
Panorama: /Users/raphaelromero/anaconda/lib/libQt5Widgets.5.6.0.dylib
Panorama: /Users/raphaelromero/anaconda/lib/libQt5Gui.5.6.0.dylib
Panorama: /Users/raphaelromero/anaconda/lib/libQt5Core.5.6.0.dylib
Panorama: CMakeFiles/Panorama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Panorama"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Panorama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Panorama.dir/build: Panorama

.PHONY : CMakeFiles/Panorama.dir/build

CMakeFiles/Panorama.dir/requires: CMakeFiles/Panorama.dir/Panorama.cpp.o.requires

.PHONY : CMakeFiles/Panorama.dir/requires

CMakeFiles/Panorama.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Panorama.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Panorama.dir/clean

CMakeFiles/Panorama.dir/depend:
	cd /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial /Users/raphaelromero/Documents/mva/3Dcompvis/Panorama_Initial/CMakeFiles/Panorama.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Panorama.dir/depend

