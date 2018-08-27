"""Detect OS"""
from platform import system
import sys

os_name = "UNKNOWN"

def detect_os(bypass=False):
    """Check OS, as multiprocessing may not work properly on Windows and macOS"""
    if bypass is True:
        return

    if isWindows():
        print("It seems that you are running this code from {}, on which the Python multiprocessing may not work properly. Consider running this code on Linux.".format(os_name))
    else:
        rosPath = '/opt/ros/kinetic/lib/python2.7/dist-packages'
        if rosPath in sys.path:
            sys.path.remove(rosPath)
        print("Linux is fine! Python multiprocessing works.")

def isWindows():
    global os_name
    if os_name in ["UNKNOWN"]:
        os_name = system()
    return os_name in ['Windows']
