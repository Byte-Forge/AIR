set(TARGET_NAME air)

#Find SFML
set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2.3 COMPONENTS graphics system REQUIRED)

#Find Boost
find_package(Boost 1.61 COMPONENTS filesystem REQUIRED)

#Create Target
add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} ${Boost_LIBRARIES} ${SFML_LIBRARIES})
target_include_directories(${TARGET_NAME} PRIVATE ${Boost_INCLUDE_DIRS} ${SFML_INCLUDE_DIRS})