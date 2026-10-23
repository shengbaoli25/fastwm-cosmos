"""A persisted RS485 configuration does not prove the RTU service is ready."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'deploy'))
from realman_deploy.gripper import Ctag2F120Worker


class GripperSetupTests(unittest.TestCase):
    def setup_pair(self):
        worker = Ctag2F120Worker({})
        sdk = Mock()
        sdk.rm_get_current_arm_state.return_value = (0, {'err': {'err': ['0']}})
        sdk.rm_get_tool_voltage.return_value = (0, 3)
        sdk.rm_get_tool_rs485_mode.return_value = dict(
            return_code=0, mode=1, baudrate=115200, timeout=2)
        sdk.rm_set_modbus_mode.return_value = 0
        registers = {0x0610: [0, 0, 0], 0x0404: [0, 0, 0, 12000],
                     0x0301: [0, 0, 0, 3000], 0x0100: [1, 0, 0],
                     0x0104: [30, 50, 100, 100], 0x060D: [0, 2401, 0]}

        def read(_sdk, _factory, address, count):
            sdk.rm_set_modbus_mode.assert_called_once_with(1, 115200, 2)
            return registers[address][:count]

        worker._read = Mock(side_effect=read)
        return worker, sdk, registers

    @patch('realman_deploy.gripper.time.sleep')
    def test_matching_settings_still_reinitialize_before_reading(self, sleep):
        worker, sdk, _ = self.setup_pair()
        lower, upper, opening = worker._setup(sdk, Mock())
        self.assertEqual((lower, upper), (0, 12000))
        self.assertAlmostEqual(opening, 1 - 2401 / 12000)
        sleep.assert_called_once_with(.3)
        sdk.rm_set_tool_voltage.assert_not_called()
        sdk.rm_write_single_register.assert_not_called()
        sdk.rm_write_registers.assert_not_called()

    @patch('realman_deploy.gripper.time.sleep')
    def test_failed_bus_initialization_stops_before_register_access(self, _sleep):
        worker, sdk, _ = self.setup_pair()
        sdk.rm_set_modbus_mode.return_value = 1
        with self.assertRaisesRegex(RuntimeError, '初始化末端 Modbus RTU'):
            worker._setup(sdk, Mock())
        worker._read.assert_not_called()
        sdk.rm_write_registers.assert_not_called()

    @patch('realman_deploy.gripper.time.sleep')
    def test_alarm_and_disabled_feedback_still_prevent_readiness(self, _sleep):
        for address, values, error in [(0x0610, [0, 0, 1], '夹爪报警'),
                                       (0x0100, [0, 0, 0], '未使能')]:
            with self.subTest(address=address):
                worker, sdk, registers = self.setup_pair()
                registers[address] = values
                with self.assertRaisesRegex(RuntimeError, error):
                    worker._setup(sdk, Mock())
                sdk.rm_write_single_register.assert_not_called()
                sdk.rm_write_registers.assert_not_called()
