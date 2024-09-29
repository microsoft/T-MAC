## Android Cross Compilation Guidance

### Pre-requisites

Install platform-tools and ndk (verified with version 26.1) from [Android Studio](https://developer.android.com/studio) or [command line tools](https://developer.android.com/studio#command-line-tools-only). Please make sure that `adb` can be found in PATH and set `NDK_HOME`.

> Using verified NDK version is recommended. TVM has stringent requirements for Clang that comes with the NDK. If you opt for Option.2 or Option.3, we recommend using Clang version 17 shipped with NDK version 26.

For example, in my PC:

```
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
export NDK_HOME="$HOME/Library/Android/sdk/ndk/26.1.10909125"
```

**There are three options to cross-compile T-MAC for Android, from simple to complex**:

### Option.1: Use Prebuilt Kernels

Using prebuilt kernels is the simplest solution.

```
python tools/run_pipeline.py -o ~/Downloads/test_models/llama-2-7b-eqat-w2g128-gptq -m llama-2-7b-2bit -d android -ndk $NDK_HOME -u
```

Please note these arguments:
- `-as`, `--adb_serial`: If there are multiple ADB devices connected to your host computer, you need to specify it according to results of `adb devices -l`.
- `-rd`, `--remote_dir`: Our binaries and models are pushed to `/data/local/tmp` for execution. Alter this argument to change the directory.

Here, we specify `-u` to use the prebuilt kernels. The performance may not be optimal.

### Option.2: Cross Compilation without Tuning

```
cd $NDK_HOME/build/tools
python make_standalone_toolchain.py --arch arm64 --install-dir /opt/android-toolchain-arm64
export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/clang++
# Back to T-MAC root dir
python tools/run_pipeline.py -o ~/Downloads/test_models/llama-2-7b-eqat-w2g128-gptq -m llama-2-7b-2bit -d android -ndk $NDK_HOME -dt
```

Here, we specify `-dt` to disable tuning. The performance may not be optimal.

### Option.3: Tuning (experimental)

Install TVM RPC APK:

```
# Back to T-MAC root dir
adb install deploy/tvmrpc-release.apk
```

Start RPC tracker:

```
python -m tvm.exec.rpc_tracker
```

Connect to the tracker in the TVM RPC APK by setting the fields:

- Address: Make sure the Android and your host PC are in the same network (e.g., wlan). Type the IP address of your host PC here
- Port: 9190
- Key: android

Then toggle on `Enable RPC`.

Verify the RPC setup with:
```
python -m tvm.exec.query_rpc_tracker
```

The setup is successful if you get something like this:

```
Tracker address 0.0.0.0:9190

Server List
------------------------------
server-address           key
------------------------------
   192.168.67.86:5001    server:android
------------------------------

Queue Status
-------------------------------
key       total  free  pending
-------------------------------
android   1      1     0      
-------------------------------
```

Finally:

```
cd $NDK_HOME/build/tools
python make_standalone_toolchain.py --arch arm64 --install-dir /opt/android-toolchain-arm64
export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/clang++
# Back to T-MAC root dir
python tools/run_pipeline.py -o ~/Downloads/test_models/llama-2-7b-eqat-w2g128-gptq -m llama-2-7b-2bit -d android -ndk $NDK_HOME
```

### Trouble-shooting

1. No device listed `python -m tvm.exec.query_rpc_tracker`

    Make sure your firewall isn't blocking incoming connections for Python. Or close your firewall.

2. `Error in RPC Tracker: [Errno 54] Connection reset by peer`

    Make sure the TVM RPC is activated and not brought to background during tuning.
