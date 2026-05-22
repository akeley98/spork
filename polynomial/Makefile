CXX      := clang++-19
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

demo: demo.cpp polynomial.hpp
	$(CXX) $(CXXFLAGS) demo.cpp -o demo

run: demo
	./demo

clean:
	rm -f demo

.PHONY: run clean
