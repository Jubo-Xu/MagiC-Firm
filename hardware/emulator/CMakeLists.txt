cmake_minimum_required(VERSION 3.20)
project(control_system_emulator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(SystemCLanguage CONFIG REQUIRED)

# Header-only platform layer (grows into a compiled lib once we add src modules).
add_library(emu_platform INTERFACE)
target_include_directories(emu_platform INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_features(emu_platform INTERFACE cxx_std_17)
target_link_libraries(emu_platform INTERFACE SystemC::systemc)

# Generic components (lib/) + detector-construct design modules, one static lib.
file(GLOB DC_SRC CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/control_system/detector_construct/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/control_system/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/control_system/cultiv_control/*.cpp)
add_library(emu_dc STATIC ${DC_SRC})
target_link_libraries(emu_dc PUBLIC emu_platform)

# Simulation executable.
add_executable(emu src/main.cpp)
target_link_libraries(emu PRIVATE emu_platform)

enable_testing()
add_executable(test_measurement_sync tests/control_system/detector_construct/test_measurement_sync.cpp)
target_link_libraries(test_measurement_sync PRIVATE emu_dc)
add_test(NAME measurement_sync COMMAND test_measurement_sync)

add_executable(test_kernel tests/control_system/detector_construct/test_kernel.cpp)
target_link_libraries(test_kernel PRIVATE emu_dc)
add_test(NAME kernel COMMAND test_kernel)

add_executable(test_raw_selector tests/control_system/detector_construct/test_raw_selector.cpp)
target_link_libraries(test_raw_selector PRIVATE emu_dc)
add_test(NAME raw_selector COMMAND test_raw_selector)

add_executable(test_output_sync tests/control_system/detector_construct/test_output_sync.cpp)
target_link_libraries(test_output_sync PRIVATE emu_dc)
add_test(NAME output_sync COMMAND test_output_sync)

add_executable(test_root_output_sync tests/control_system/detector_construct/test_root_output_sync.cpp)
target_link_libraries(test_root_output_sync PRIVATE emu_dc)
add_test(NAME root_output_sync COMMAND test_root_output_sync)

add_executable(test_postselect tests/control_system/detector_construct/test_postselect.cpp)
target_link_libraries(test_postselect PRIVATE emu_dc)
add_test(NAME postselect COMMAND test_postselect)

add_executable(test_detector_pass tests/control_system/detector_construct/test_detector_pass.cpp)
target_link_libraries(test_detector_pass PRIVATE emu_dc)
add_test(NAME detector_pass COMMAND test_detector_pass)

add_executable(test_regfile_rom tests/lib/test_regfile_rom.cpp)
target_link_libraries(test_regfile_rom PRIVATE emu_dc)
add_test(NAME regfile_rom COMMAND test_regfile_rom)

add_executable(test_banked_ram tests/lib/test_banked_ram.cpp)
target_link_libraries(test_banked_ram PRIVATE emu_dc)
add_test(NAME banked_ram COMMAND test_banked_ram)

add_executable(test_sync_rom tests/lib/test_sync_rom.cpp)
target_link_libraries(test_sync_rom PRIVATE emu_dc)
add_test(NAME sync_rom COMMAND test_sync_rom)

add_executable(test_instr_decode tests/control_system/cultiv_control/test_instr_decode.cpp)
target_link_libraries(test_instr_decode PRIVATE emu_dc)
add_test(NAME instr_decode COMMAND test_instr_decode)

add_executable(test_instr_sequencer tests/control_system/cultiv_control/test_instr_sequencer.cpp)
target_link_libraries(test_instr_sequencer PRIVATE emu_dc)
add_test(NAME instr_sequencer COMMAND test_instr_sequencer)

add_executable(test_instr_unpack tests/control_system/cultiv_control/test_instr_unpack.cpp)
target_link_libraries(test_instr_unpack PRIVATE emu_dc)
add_test(NAME instr_unpack COMMAND test_instr_unpack)

add_executable(test_physical_mmio tests/control_system/cultiv_control/test_physical_mmio.cpp)
target_link_libraries(test_physical_mmio PRIVATE emu_dc)
add_test(NAME physical_mmio COMMAND test_physical_mmio)

add_executable(test_board_control tests/control_system/cultiv_control/test_board_control.cpp)
target_link_libraries(test_board_control PRIVATE emu_dc)
add_test(NAME board_control COMMAND test_board_control)

add_executable(test_stim_readout tests/control_system/test_stim_readout.cpp)
target_link_libraries(test_stim_readout PRIVATE emu_dc)
add_test(NAME stim_readout COMMAND test_stim_readout)

add_executable(test_drain_aggregator tests/control_system/cultiv_control/test_drain_aggregator.cpp)
target_link_libraries(test_drain_aggregator PRIVATE emu_dc)
add_test(NAME drain_aggregator COMMAND test_drain_aggregator)

add_executable(test_board_loader tests/control_system/detector_construct/test_board_loader.cpp)
target_link_libraries(test_board_loader PRIVATE emu_dc)
# (run manually with the example dir; not registered as a ctest since it needs the compiled results)

add_executable(test_control_board_loader tests/control_system/test_control_board_loader.cpp)
target_link_libraries(test_control_board_loader PRIVATE emu_dc)

add_executable(test_control_board_stim tests/control_system/test_control_board_stim.cpp)
target_link_libraries(test_control_board_stim PRIVATE emu_dc)
# (run manually with the example dir; whole-system loop test)
# (run manually with the example dir; not registered as a ctest since it needs the compiled results)

add_executable(test_dcb_stim tests/control_system/detector_construct/test_dcb_stim.cpp)
target_link_libraries(test_dcb_stim PRIVATE emu_dc)
# (run manually: ./test_dcb_stim <example_dir> [--shots N --input-gap G --shot-gap G])
