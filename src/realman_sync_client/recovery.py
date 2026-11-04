"""Explicit allowlist for discarding a failed cycle and collecting afresh."""
from urllib.error import HTTPError, URLError


_TRANSIENT = (
    'Controller response timeout', 'Controller send deadline exceeded',
    '夹爪通信曾中断', '夹爪运动反馈过期', '夹爪使能反馈过期',
    '夹爪运动反馈未就绪', '夹爪尚未就绪',
    '相机与机械臂状态时间差超限', '两路相机时间差超限', '相机画面过期',
    '等待相机新帧', '缺少状态锚点之前的画面', '画面过期 (状态差',
    'RM75 状态/力反馈过期', 'UDP 反馈过期',
    '力滤波预热中', '高频力速率不是约 200 Hz',
    '推理期间力滤波器重置，已丢弃 action',
    'force filter epoch changed while awaiting this action result',
    'robot/camera observation is stale', 'latest observation is stale',
    '30 Hz observation history has a gap', '30 Hz observation grid overrun',
    'timed out waiting for fresh 30 Hz observation history',
    '30 Hz hardware command period overrun; refusing a catch-up burst',
    'server is not ready for inference',
)
_HARD = ('急停', '缓停失败', '缓停超时', '未使能', '失能', '报警',
         '控制器或关节反馈报错', 'soft-limit', 'non-finite', '不匹配',
         '坐标系', '身份', '指令失败', 'command mismatch', 'explicit reconnect')


def transient_failure(error):
    """Unknown failures do not authorize automatic robot re-arming."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = getattr(error, '__cause__', None)
    if any(getattr(item, 'recoverable', None) is False for item in chain):
        return False
    if any(any(word in str(item) for word in _HARD) for item in chain):
        return False
    for item in chain:
        if isinstance(item, HTTPError):
            return item.code in (408, 429, 502, 503, 504)
    return any(isinstance(item, (TimeoutError, ConnectionError, URLError))
               or any(word in str(item) for word in _TRANSIENT) for item in chain)
