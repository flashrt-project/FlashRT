# pi0.5 policy inference on a FlashRT TensorRT engine (overlay for TensorRT Edge-LLM).
add_executable(pi05_policy_inference pi05_policy_inference.cpp)
target_link_libraries(pi05_policy_inference PRIVATE edgellmCore exampleUtils commonLibraryExt)
target_include_directories(pi05_policy_inference PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
                                                         ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(pi05_policy_inference)
