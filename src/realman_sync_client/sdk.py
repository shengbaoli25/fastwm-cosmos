"""Initialize RealMan's process-global native runtime once, in UDP mode."""
import threading

_lock = threading.Lock()
_initialized = False


def create_sdk():
    global _initialized
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    with _lock:
        if not _initialized:
            sdk = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            _initialized = True
            return sdk
        # A second rm_init changes the global native threads, including UDP.
        return RoboticArm()
