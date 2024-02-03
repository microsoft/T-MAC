~/Library/Android/sdk/ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ -target aarch64-linux-android21 -O3 -march=armv8.2a+fp16 test_tbl.cc
~/Library/Android/sdk/platform-tools/adb push a.out /data/local/tmp
~/Library/Android/sdk/platform-tools/adb shell taskset 112 /data/local/tmp/a.out
~/Library/Android/sdk/platform-tools/adb shell /data/local/tmp/a.out
