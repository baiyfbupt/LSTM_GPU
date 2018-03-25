echo off
set flag=%2
adb.exe push %1 /data/local/test/


if %flag% == 1 (adb.exe shell chmod 755 /data/local/test/%1;
                    adb.exe shell /data/local/test/%1)


if %flag% == 2 (adb.exe shell /data/local/test/%1)
