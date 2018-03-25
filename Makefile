CXX = aarch64-linux-android-clang++
CC = aarch64-linux-android-clang

#PROF = -fprofile-instr-generate
CXXFLAGS = -O3 -Ofast  -fexceptions -frtti -std=c++11  -pie
CXXFLAGS += -Winline
CXXFLAGS += -mfpu=neon -mfloat-abi=softfp
LDFLAGS = -Wl,-rpath AndroidARM/sysroot/usr/lib   -lGLES_mali -llog

all:
	$(CXX) $(CXXFLAGS) $(PROF) *.cpp -o lstm $(LDFLAGS)
clean:
	rm  lstm
