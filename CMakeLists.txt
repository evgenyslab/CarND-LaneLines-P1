cmake_minimum_required(VERSION 3.6)

project(UND-P01)

enable_testing()

# Find Python and Virtualenv. We don't actually use the output of the
# find_package, but it'll give nicer errors.
find_package(PythonInterp 2.7 REQUIRED)
find_program(VIRTUALENV virtualenv)
if(NOT VIRTUALENV)
    message(FATAL_ERROR "Could not find `virtualenv` in PATH")
endif()
#set(VIRTUALENV ${VIRTUALENV} -p python2.7)
set(PYTHON_EXECUTABLE ~/.virtualenvs/cv/bin/python2.7)
set(PYTHON_INCLUDE_DIR ~/.virtualenvs/cv/include/python2.7)
MESSAGE(blah)

