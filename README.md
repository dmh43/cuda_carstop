# CUDA/C++11 cmake starter with google test 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A cross-platform CUDA/C++11 starter project with google test support.

# Build

On Linux/Unix, to build and make the test:

    $ mkdir build && cd $_
    $ cmake ..
    $ make

By default, the makefiles will build the library, executable and tests. The commands

    $ ./test/deepgreen_tests

...will run the tests.

On Windows, you can use cmake to generate Visual Studio build files with
the same 'cmake ..' command.

By default, the project will be built in RELEASE mode, use

    $ cmake .. -DCMAKE_BUILD_TYPE=DEBUG

to build in DEBUG mode.

See the CMakeLists.txt file to see all the options.

# License

[MIT](http://opensource.org/licenses/MIT)

