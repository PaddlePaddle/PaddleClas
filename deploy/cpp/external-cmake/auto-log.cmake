FIND_PACKAGE(Git REQUIRED)
INCLUDE(FetchContent)

SET(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party")

FetchContent_Declare(
  extern_Autolog
  PREFIX autolog
  GIT_REPOSITORY https://github.com/LDOUBLEV/AutoLog.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(extern_Autolog)