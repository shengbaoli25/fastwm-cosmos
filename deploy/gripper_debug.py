#!/usr/bin/env python3
"""CTAG2F120 debugging through the running local inference service."""
from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'connect', 'enable', 'disable',
                                         'clear_error', 'open', 'close', 'move'])
    parser.add_argument('--opening', type=float, help='move 的目标开度百分比，0=闭合，100=打开')
    parser.add_argument('--speed', type=int, help='移动速度，1..100')
    parser.add_argument('--torque', type=int, help='移动力度，1..100')
    parser.add_argument('--url', default='http://127.0.0.1:8765', help='本机推理服务地址')
    parser.add_argument('--cached', action='store_true', help='status 仅读取缓存，不请求寄存器读取')
    args = parser.parse_args(argv)
    url = urlparse(args.url)
    if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or url.username or url.password or url.path not in {'', '/'} or url.query or url.fragment):
        parser.error('--url 必须为本机 HTTP 服务根地址')
    if args.cached and args.action != 'status':
        parser.error('--cached 仅用于 status')
    if args.action == 'move':
        if args.opening is None or not 0 <= args.opening <= 100:
            parser.error('move 需要 --opening 0..100')
    elif args.opening is not None:
        parser.error('--opening 仅用于 move')
    for key in ('speed', 'torque'):
        value = getattr(args, key)
        if value is not None and (args.action not in {'open', 'close', 'move'} or not 1 <= value <= 100):
            parser.error('--' + key + ' 只用于移动，范围为 1..100')
    body = {'action': args.action}
    if args.opening is not None:
        body['opening'] = args.opening / 100
    for key in ('speed', 'torque'):
        if getattr(args, key) is not None:
            body[key] = getattr(args, key)
    request = Request(args.url.rstrip('/') + '/api/gripper',
                      data=None if args.cached else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=20) as response:
            print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
        return 0
    except HTTPError as exc:
        print(exc.read().decode('utf-8', errors='replace'), file=sys.stderr)
    except (URLError, TimeoutError) as exc:
        print(f'夹爪请求失败：{exc}。未自动重试；请先读取状态确认。', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
