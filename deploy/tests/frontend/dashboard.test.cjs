'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createDashboard} = require('../../realman_deploy/static/app.js');

const html = fs.readFileSync(path.join(__dirname, '../../realman_deploy/static/index.html'), 'utf8');
function fixture(overrides = {}) {
  const elements = new Map([...html.matchAll(/id="([^"]+)"([^>]*)/g)].map((match) => [match[1], {
    textContent: '', value: match[2].match(/value="([^"]*)"/)?.[1] || '', disabled: false, hidden: false, checked: false, className: '', children: [],
    replaceChildren() { this.children = []; this.value = ''; }, append(child) { this.children.push(child); },
  }]));
  const state = {status: {state: 'ready', epoch: 1, model_ready: true, robot_ready: true,
    estop_latched: overrides.state === 'estop', estop_reset_busy: false, estop_reset_supported: true,
    cameras_ready: true, force_ready: true, gripper_ready: true, inference_busy: false,
    sampling_profile: '161616', execution_mode: 'synchronous', execution_phase: 'ready',
    gripper_debug: {supported: true, connected: true, ready: true, enabled: true, alarm: 0, alarm_hex: '0x0000', alarm_labels: [],
      opening: 0.5, position: 500, error: null, busy: false, moving: false, operation: null}, gripper_debug_busy: false,
    cuda: {available: true, device: 'Test GPU', total_bytes: 24000000000}, ...overrides},
    settings: {sampling_profile: '161616', execution_mode: 'synchronous', supported: true}, calls: [], handlers: {}, timers: new Map(), timerDurations: new Map(), nextTimer: 0};
  const fetch = async (url, options) => {
    state.calls.push({url, ...options});
    if (state.handlers[url]) return state.handlers[url](options);
    let result;
    if (url === '/api/status') result = {...state.status};
    else if (url === '/api/settings' && options.method === 'GET') result = {...state.settings};
    else if (url === '/api/settings') {
      state.settings = {...state.settings, ...JSON.parse(options.body)};
      Object.assign(state.status, state.settings, {model_ready: false}); result = {...state.settings};
    } else if (url === '/api/checkpoints') result = {directory: '/checkpoints', checkpoints: [{id: 'step10000', label: 'Step 10000', installed: true}]};
    else if (url === '/api/gripper') {
      if (options.method === 'POST') {
        const payload = JSON.parse(options.body), g = state.status.gripper_debug;
        g.operation = payload.action;
        if (payload.action === 'connect') g.connected = true;
        if (payload.action === 'enable') g.enabled = true;
        if (payload.action === 'disable') g.enabled = false;
        if (payload.action === 'clear_error') Object.assign(g, {alarm: 0, alarm_hex: '0x0000', alarm_labels: [], enabled: false});
        if (payload.action === 'move') g.opening = payload.opening;
        if (payload.action === 'open') g.opening = 1;
        if (payload.action === 'close') g.opening = 0;
      }
      result = {...state.status.gripper_debug};
    }
    else if (url === '/api/estop') { Object.assign(state.status, {state: 'estop', estop_latched: true, epoch: state.status.epoch + 1}); result = {...state.status}; }
    else if (url === '/api/estop/reset') { Object.assign(state.status, {state: 'paused', estop_latched: false, estop_reset_busy: false, epoch: state.status.epoch + 1}); result = {...state.status}; }
    else if (url === '/api/pause') { state.status.state = 'paused'; result = {state: 'paused'}; }
    else result = {};
    return {ok: true, status: 200, json: async () => result};
  };
  const dashboard = createDashboard({document: {getElementById: id => { assert(elements.has(id), `Missing #${id}`); return elements.get(id); },
    createElement: () => ({textContent: '', value: ''})}, fetch,
    setTimeout: (callback, duration) => { state.timers.set(++state.nextTimer, callback); state.timerDurations.set(state.nextTimer, duration); return state.nextTimer; },
    clearTimeout: id => state.timers.delete(id), setInterval: () => 1, clearInterval: () => {}, now: () => new Date('2026-10-23T00:00:00Z')});
  return {dashboard, state, $: id => elements.get(id), choose(id, value) { elements.get(id).value = value; elements.get(id).onchange(); },
    posts(url) { return state.calls.filter(call => call.url === url && call.method === 'POST'); }};
}
const pending = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return {promise, resolve}; };

test('4416 exposes async mode, applies supported settings, and requires model reload', async () => {
  const f = fixture(); await f.dashboard.init();
  assert.equal(f.$('executionField').hidden, true);
  f.choose('samplingProfile', '4416'); f.choose('executionSelect', 'async_relaxed');
  assert.equal(f.$('executionField').hidden, false);
  assert.equal(f.$('applySettings').disabled, false);
  assert.equal(f.$('start').disabled, true, 'unapplied draft must not silently start old settings');
  await f.$('applySettings').onclick();
  assert.deepEqual(JSON.parse(f.posts('/api/settings')[0].body), {sampling_profile: '4416', execution_mode: 'async_relaxed'});
  assert.equal(f.$('start').disabled, true, 'updated schedule invalidates loaded policy');
  assert.equal(f.$('load').disabled, false);
  f.choose('samplingProfile', '161616');
  assert.equal(f.$('executionField').hidden, true);
  assert.equal(f.$('executionSelect').value, 'synchronous');
  await f.$('applySettings').onclick();
  assert.equal(JSON.parse(f.posts('/api/settings')[1].body).execution_mode, 'synchronous');
});

test('running and in-flight inference prevent applying settings until pause is settled', async () => {
  const f = fixture({state: 'running', inference_busy: true}); await f.dashboard.init();
  assert.equal(f.$('samplingProfile').disabled, true);
  await f.$('applySettings').onclick();
  assert.equal(f.posts('/api/settings').length, 0);
  await f.$('pause').onclick();
  assert.equal(f.$('samplingProfile').disabled, true, 'paused state alone does not mean inference completed');
  f.state.status.inference_busy = false; await f.dashboard.refreshStatus();
  assert.equal(f.$('samplingProfile').disabled, false);
});

test('legacy action spaces keep inference usable but disable joint8 settings', async () => {
  const f = fixture(); f.state.settings.supported = false; await f.dashboard.init();
  assert.equal(f.$('samplingProfile').disabled, true);
  assert.equal(f.$('applySettings').disabled, true);
  assert.match(f.$('settingsHint').textContent, /仅适用于 joint8/);
  assert.equal(f.$('load').disabled, false);
  assert.equal(f.$('start').disabled, false);
});

test('emergency stop bypasses pending load and duplicate ordinary operations are rejected', async () => {
  const f = fixture(); await f.dashboard.init();
  const load = pending(); f.state.handlers['/api/select'] = () => load.promise;
  const loading = f.$('load').onclick();
  await f.$('load').onclick();
  assert.equal(f.posts('/api/select').length, 1);
  assert.equal(f.$('connect').disabled, true);
  assert.equal(f.$('estop').disabled, false);
  await f.$('estop').onclick();
  assert.equal(f.posts('/api/estop').length, 1);
  assert.equal(f.$('start').disabled, true);
  load.resolve({ok: true, status: 202, json: async () => ({state: 'loading'})});
  await loading;
  assert.equal(f.$('start').disabled, true, 'late load response cannot remove emergency latch');
  assert.equal(f.$('estopNotice').hidden, false);
});

test('pause also bypasses long ordinary operations', async () => {
  const f = fixture(); await f.dashboard.init();
  const connect = pending(); f.state.handlers['/api/connect'] = () => connect.promise;
  const connecting = f.$('connect').onclick();
  await f.$('pause').onclick();
  assert.equal(f.posts('/api/pause').length, 1);
  connect.resolve({ok: true, status: 200, json: async () => ({})}); await connecting;
});

test('failed emergency stop remains locally latched and can be retried', async () => {
  const f = fixture(); await f.dashboard.init();
  f.state.handlers['/api/estop'] = async () => { throw new Error('网络已断开'); };
  await f.$('estop').onclick();
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('connect').disabled, true);
  assert.equal(f.$('estop').disabled, false, 'failed stop remains retryable');
  assert.match(f.$('error').textContent, /急停未确认/);
  await f.dashboard.refreshStatus();
  assert.match(f.$('error').textContent, /急停未确认/);
});

test('timeout releases busy state and reports uncertainty instead of claiming success', async () => {
  const f = fixture(); await f.dashboard.init();
  f.state.handlers['/api/connect'] = options => new Promise((_, reject) => {
    options.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), {name: 'AbortError'})));
  });
  const connecting = f.$('connect').onclick();
  for (const callback of f.state.timers.values()) callback();
  await connecting;
  assert.match(f.$('error').textContent, /请求超时.*\/api\/connect/);
  assert.match(f.$('error').textContent, /服务端操作可能仍在进行/);
  assert.equal(f.$('connect').disabled, false);
  assert.equal(f.state.timers.size, 0);
});

test('lost status connection disables start and status polls do not overlap', async () => {
  const f = fixture(); await f.dashboard.init();
  const status = pending(); f.state.handlers['/api/status'] = () => status.promise;
  const initialCalls = f.state.calls.length;
  const first = f.dashboard.refreshStatus(), second = f.dashboard.refreshStatus();
  assert.equal(f.state.calls.length - initialCalls, 1);
  status.resolve({ok: false, status: 503, json: async () => ({error: '服务不可用'})});
  await Promise.all([first, second]);
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('load').disabled, true);
  assert.equal(f.$('estop').disabled, false);
  assert.match(f.$('error').textContent, /服务不可用/);
});

test('checkpoint labels are literal text and async latency reports possible waiting', async () => {
  const f = fixture({execution_mode: 'async_relaxed', sampling_profile: '4416', latency_ms: 1500});
  f.state.handlers['/api/checkpoints'] = async () => ({ok: true, status: 200, json: async () => ({directory: '/checkpoints',
    checkpoints: [{id: 'remote', label: '<img onerror=alert(1)>', installed: false}]})});
  await f.dashboard.init();
  assert.equal(f.$('checkpoint').children[0].textContent, '<img onerror=alert(1)> · 待下载');
  assert.equal(f.$('load').disabled, true);
  assert.equal(f.$('download').disabled, false);
  assert.match(f.$('modeHint').textContent, /可能出现等待间隔/);
  assert.equal(f.$('latency').textContent, '1500.0 ms');
});


test('missing settings never enables control with an unconfirmed schedule', async () => {
  const f = fixture();
  f.state.handlers['/api/settings'] = async () => ({ok: false, status: 503, json: async () => ({error: '设置读取失败'})});
  await f.dashboard.init();
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('load').disabled, true);
  assert.equal(f.$('estop').disabled, false);
  assert.match(f.$('error').textContent, /设置读取失败/);
});

test('gripper connects independently without a loaded model, GPU, or cameras', async () => {
  const f = fixture({state: 'idle', model_ready: false, cameras_ready: false, robot_ready: false, cuda: {available: false},
    gripper_debug: {supported: true, connected: false, enabled: false, alarm: null, busy: false}});
  await f.dashboard.init();
  assert.equal(f.$('gripperConnect').disabled, false);
  for (const id of ['gripperRead', 'gripperEnable', 'gripperDisable', 'gripperClear', 'gripperOpen', 'gripperClose', 'gripperMove']) {
    assert.equal(f.$(id).disabled, true, id);
    await f.$(id).onclick();
  }
  assert.equal(f.posts('/api/gripper').length, 0);
  await f.$('gripperConnect').onclick();
  assert.deepEqual(JSON.parse(f.posts('/api/gripper')[0].body), {action: 'connect'});
  assert.equal(f.$('gripperConnection').textContent, '已连接');
  assert.equal(f.$('gripperConnect').textContent, '重新连接夹爪');
});

test('gripper commands send normalized openings and explicit speed and force', async () => {
  const f = fixture(); await f.dashboard.init();
  assert.equal(f.$('gripperSpeed').value, '30');
  assert.equal(f.$('gripperTorque').value, '50');
  await f.$('gripperOpen').onclick();
  await f.$('gripperClose').onclick();
  f.$('gripperTarget').value = '37.5'; f.$('gripperSpeed').value = '20'; f.$('gripperTorque').value = '45';
  await f.$('gripperMove').onclick();
  assert.deepEqual(f.posts('/api/gripper').map(call => JSON.parse(call.body)), [
    {action: 'open', speed: 30, torque: 50}, {action: 'close', speed: 30, torque: 50},
    {action: 'move', opening: 0.375, speed: 20, torque: 45},
  ]);
  assert.equal(f.$('gripperOpening').textContent, '37.5%');
  assert.match(f.$('gripperFeedback').textContent, /反馈已确认/);
});

test('gripper rejects invalid inputs before sending movement', async () => {
  const f = fixture(); await f.dashboard.init();
  f.$('gripperSpeed').value = '0'; await f.$('gripperOpen').onclick();
  assert.match(f.$('gripperError').textContent, /1–100/);
  f.$('gripperSpeed').value = '30'; f.$('gripperTarget').value = '101'; await f.$('gripperMove').onclick();
  assert.match(f.$('gripperError').textContent, /0–100/);
  f.$('gripperTarget').value = ''; await f.$('gripperMove').onclick();
  assert.equal(f.posts('/api/gripper').length, 0);
});

test('alarm feedback blocks movement and clearing an alarm leaves the gripper disabled', async () => {
  const f = fixture(); Object.assign(f.state.status.gripper_debug, {alarm: 8, alarm_hex: '0x0008', alarm_labels: ['过流保护'], enabled: false});
  await f.dashboard.init();
  assert.equal(f.$('gripperOpen').disabled, true);
  assert.equal(f.$('gripperEnable').disabled, true);
  assert.equal(f.$('gripperClear').disabled, false);
  assert.match(f.$('gripperAlarm').textContent, /0x0008/);
  assert.equal(f.$('gripperAlarmLabels').textContent, '过流保护');
  await f.$('gripperClear').onclick();
  assert.deepEqual(JSON.parse(f.posts('/api/gripper')[0].body), {action: 'clear_error'});
  assert.equal(f.$('gripperOpen').disabled, true);
  assert.equal(f.$('gripperEnable').disabled, false);
  assert.match(f.$('gripperFeedback').textContent, /保持失能/);
  await f.$('gripperEnable').onclick();
  assert.equal(f.$('gripperOpen').disabled, false);
  await f.$('gripperDisable').onclick();
  assert.equal(f.$('gripperOpen').disabled, true);
});

test('running, loading, stopping, inference in flight, and estop lock every gripper command', async () => {
  for (const overrides of [{state: 'running'}, {state: 'loading'}, {state: 'stopping'}, {state: 'paused', inference_busy: true}, {state: 'estop'}]) {
    const f = fixture(overrides); await f.dashboard.init();
    for (const id of ['gripperConnect', 'gripperRead', 'gripperEnable', 'gripperDisable', 'gripperClear', 'gripperOpen', 'gripperClose', 'gripperMove']) {
      assert.equal(f.$(id).disabled, true, `${overrides.state}: ${id}`);
      await f.$(id).onclick();
    }
    assert.equal(f.posts('/api/gripper').length, 0);
  }
});

test('gripper operation excludes inference operations but leaves pause and estop available', async () => {
  const f = fixture(); await f.dashboard.init();
  f.choose('samplingProfile', '4416');
  const command = pending(); f.state.handlers['/api/gripper'] = () => command.promise;
  const opening = f.$('gripperOpen').onclick();
  assert([...f.state.timerDurations.values()].includes(20000));
  for (const id of ['load', 'connect', 'start', 'applySettings', 'gripperRead', 'gripperOpen', 'gripperConnect']) {
    assert.equal(f.$(id).disabled, true, id);
    await f.$(id).onclick();
  }
  assert.equal(f.posts('/api/gripper').length, 1);
  assert.equal(f.$('samplingProfile').disabled, true);
  assert.equal(f.$('pause').disabled, false); assert.equal(f.$('estop').disabled, false);
  await f.$('pause').onclick();
  command.resolve({ok: true, status: 200, json: async () => ({...f.state.status.gripper_debug, opening: 1})});
  await opening;
  assert.equal(f.posts('/api/pause').length, 1);
  assert.equal(f.posts('/api/select').length, 0); assert.equal(f.posts('/api/connect').length, 0);
});

test('server gripper busy flag and normal pending load both block debug commands', async () => {
  const f = fixture({gripper_debug_busy: true}); await f.dashboard.init();
  assert.equal(f.$('gripperOpen').disabled, true); assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('load').disabled, true); assert.equal(f.$('connect').disabled, true);
  f.state.status.gripper_debug_busy = false; await f.dashboard.refreshStatus();
  const load = pending(); f.state.handlers['/api/select'] = () => load.promise;
  const loading = f.$('load').onclick();
  assert.equal(f.$('gripperConnect').disabled, true);
  await f.$('gripperOpen').onclick(); assert.equal(f.posts('/api/gripper').length, 0);
  load.resolve({ok: true, status: 200, json: async () => ({})}); await loading;
});

test('gripper timeout is uncertain, does not retry, and server busy stays locked', async () => {
  const f = fixture(); await f.dashboard.init();
  f.state.handlers['/api/gripper'] = options => new Promise((_, reject) => {
    options.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), {name: 'AbortError'})));
  });
  const opening = f.$('gripperOpen').onclick();
  f.state.status.gripper_debug_busy = true;
  for (const callback of f.state.timers.values()) callback();
  await opening;
  assert.equal(f.posts('/api/gripper').length, 1);
  assert.match(f.$('gripperError').textContent, new RegExp('请求超时.*/api/gripper'));
  assert.match(f.$('gripperFeedback').textContent, /未确认/);
  assert.equal(f.$('start').disabled, true); assert.equal(f.$('gripperOpen').disabled, true);
  assert.equal(f.$('estop').disabled, false);
});

test('gripper success never clears an unconfirmed stop error and errors survive status polls', async () => {
  const f = fixture(); await f.dashboard.init();
  f.state.handlers['/api/pause'] = async () => { throw new Error('暂停链路异常'); };
  await f.$('pause').onclick();
  await f.$('gripperRead').onclick();
  assert.match(f.$('error').textContent, /暂停未确认/);
  assert.equal(f.$('gripperFeedback').textContent, '已读取夹爪状态。');
  f.state.handlers['/api/gripper'] = async () => ({ok: false, status: 409, json: async () => ({error: '夹爪驱动未确认使能'})});
  await f.$('gripperOpen').onclick();
  await f.dashboard.refreshStatus();
  assert.match(f.$('gripperError').textContent, /夹爪驱动未确认使能/);
  assert.match(f.$('error').textContent, /暂停未确认/);
});

test('cached server command errors remain visible in the gripper panel', async () => {
  const f = fixture(); f.state.status.gripper_debug.command_error = '末端 Modbus 读取失败';
  await f.dashboard.init();
  assert.match(f.$('gripperError').textContent, /末端 Modbus 读取失败/);
  assert.equal(f.$('gripperDebugBadge').textContent, '需检查');
  await f.dashboard.refreshStatus();
  assert.match(f.$('gripperError').textContent, /末端 Modbus 读取失败/);
});

test('cross-page hardware connection locks debug and inference operations until it completes', async () => {
  const f = fixture({hardware_connecting: true}); await f.dashboard.init();
  for (const id of ['start', 'load', 'connect', 'gripperConnect', 'gripperRead', 'gripperOpen', 'gripperDisable', 'gripperClear']) {
    assert.equal(f.$(id).disabled, true, id);
    await f.$(id).onclick();
  }
  assert.equal(f.posts('/api/gripper').length, 0);
  assert.equal(f.$('samplingProfile').disabled, true);
  assert.equal(f.$('pause').disabled, false); assert.equal(f.$('estop').disabled, false);
  assert.match(f.$('gripperGuard').textContent, /硬件连接进行中/);
  f.state.status.hardware_connecting = false; await f.dashboard.refreshStatus();
  assert.equal(f.$('gripperOpen').disabled, false); assert.equal(f.$('start').disabled, false);
});

test('disabled device feedback confirms disable and clear commands while retaining device errors', async () => {
  const f = fixture(); await f.dashboard.init();
  f.state.handlers['/api/gripper'] = async options => {
    const action = JSON.parse(options.body).action;
    const result = {...f.state.status.gripper_debug, enabled: false, ready: false, alarm: 0,
      error: '夹爪未使能', operation: action};
    f.state.status.gripper_debug = result;
    return {ok: true, status: 200, json: async () => result};
  };
  await f.$('gripperDisable').onclick();
  assert.match(f.$('gripperFeedback').textContent, /失能完成/);
  assert.match(f.$('gripperError').textContent, /夹爪未使能/);
  await f.$('gripperClear').onclick();
  assert.match(f.$('gripperFeedback').textContent, /报警已清除.*保持失能/);
  assert.match(f.$('gripperError').textContent, /夹爪未使能/);
  assert.equal(f.$('gripperOpen').disabled, true);
});

test('connection and status reading may succeed with alarms without claiming motion success', async () => {
  const f = fixture(); await f.dashboard.init();
  f.state.handlers['/api/gripper'] = async options => {
    const result = {...f.state.status.gripper_debug, ready: false, alarm: 8, alarm_hex: '0x0008',
      alarm_labels: ['过流保护'], error: '夹爪报警', operation: JSON.parse(options.body).action};
    f.state.status.gripper_debug = result;
    return {ok: true, status: 200, json: async () => result};
  };
  await f.$('gripperConnect').onclick();
  assert.match(f.$('gripperFeedback').textContent, /连接夹爪完成/);
  assert.match(f.$('gripperError').textContent, /夹爪报警/);
  await f.$('gripperRead').onclick();
  assert.equal(f.$('gripperFeedback').textContent, '已读取夹爪状态。');
  assert.equal(f.$('gripperOpen').disabled, true);

  Object.assign(f.state.status.gripper_debug, {ready: true, alarm: 0, enabled: true, error: null});
  await f.dashboard.refreshStatus();
  await f.$('gripperOpen').onclick();
  assert.equal(f.$('gripperFeedback').textContent, '打开未确认。');
  assert.match(f.$('gripperError').textContent, /夹爪报警/);
});

test('cancelled or unconfirmed command feedback never reports success', async () => {
  for (const feedback of [{command_completed: false}, {operation_result: {cancelled: true, message: '操作已取消'}},
    {operation_result: {status: 'stopped'}}, {operation_result: {stop_confirmed: false}}, {operation_result: '操作被暂停中断'}]) {
    const f = fixture(); await f.dashboard.init();
    f.state.handlers['/api/gripper'] = async () => ({ok: true, status: 200,
      json: async () => ({...f.state.status.gripper_debug, ...feedback})});
    await f.$('gripperOpen').onclick();
    assert.equal(f.$('gripperFeedback').textContent, '打开未确认。');
    assert.equal(f.$('gripperError').hidden, false);
  }
});

test('enable requires ready feedback even if its enabled bit is set', async () => {
  const f = fixture(); f.state.status.gripper_debug.enabled = false; await f.dashboard.init();
  f.state.handlers['/api/gripper'] = async () => ({ok: true, status: 200, json: async () => ({
    ...f.state.status.gripper_debug, enabled: true, ready: false, error: '夹爪状态尚未就绪', command_completed: true,
  })});
  await f.$('gripperEnable').onclick();
  assert.equal(f.$('gripperFeedback').textContent, '使能未确认。');
  assert.match(f.$('gripperError').textContent, /夹爪状态尚未就绪/);
});

test('estop recovery requires explicit confirmation and only returns to paused', async () => {
  const f = fixture(); await f.dashboard.init();
  assert.equal(f.$('estopReset').disabled, true);
  await f.$('estopReset').onclick();
  assert.equal(f.posts('/api/estop/reset').length, 0);
  await f.$('estop').onclick();
  assert.equal(f.$('estopResetConfirmed').disabled, false);
  assert.equal(f.$('estopReset').disabled, true);
  assert.match(f.$('estopResetHint').textContent, /先解除.*物理急停.*现场/);
  f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
  assert.equal(f.$('estopReset').disabled, false);
  await f.$('estopReset').onclick();
  assert.deepEqual(JSON.parse(f.posts('/api/estop/reset')[0].body), {confirmed: true});
  assert.equal(f.state.status.state, 'paused');
  assert.equal(f.$('estopNotice').hidden, true);
  assert.equal(f.$('estopResetConfirmed').checked, false);
  assert.match(f.$('estopResetFeedback').textContent, /保持暂停.*手动开始/);
  assert.equal(f.posts('/api/start').length, 0);
  assert.equal(f.posts('/api/connect').length, 0);
  assert.equal(f.posts('/api/gripper').length, 0);
});

test('legacy servers cannot falsely offer or confirm estop recovery', async () => {
  for (const supported of [undefined, false]) {
    const f = fixture({state: 'estop', estop_reset_supported: supported}); await f.dashboard.init();
    assert.equal(f.$('estopResetConfirmed').disabled, true);
    f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
    assert.equal(f.$('estopReset').disabled, true);
    await f.$('estopReset').onclick();
    assert.equal(f.posts('/api/estop/reset').length, 0);
    assert.match(f.$('estopResetHint').textContent, /尚未提供.*恢复接口/);
  }
});

test('failed estop recovery preserves the latch, error, and explicit retry', async () => {
  const f = fixture({state: 'estop'}); await f.dashboard.init();
  f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
  f.state.handlers['/api/estop/reset'] = async () => ({ok: false, status: 409, json: async () => ({error: '机械臂停止状态未确认'})});
  await f.$('estopReset').onclick();
  assert.equal(f.$('estopNotice').hidden, false);
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('gripperOpen').disabled, true);
  assert.equal(f.$('estopReset').disabled, false);
  assert.match(f.$('error').textContent, /恢复急停未确认.*停止状态未确认/);
  await f.dashboard.refreshStatus();
  assert.match(f.$('error').textContent, /恢复急停未确认/);
  delete f.state.handlers['/api/estop/reset'];
  await f.$('estopReset').onclick();
  assert.equal(f.$('estopNotice').hidden, true);
  assert.equal(f.$('error').textContent, '');
});

test('recovery success requires explicit latch feedback and fresh paused status', async () => {
  for (const feedback of [{state: 'paused'}, {state: 'paused', estop_latched: true, estop_reset_busy: false},
    {state: 'paused', estop_latched: false, estop_reset_busy: true},
    {state: 'paused', estop_latched: false, estop_reset_busy: false, epoch: 9}]) {
    const f = fixture({state: 'estop'}); await f.dashboard.init();
    f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
    f.state.handlers['/api/estop/reset'] = async () => ({ok: true, status: 200, json: async () => feedback});
    await f.$('estopReset').onclick();
    assert.equal(f.$('estopNotice').hidden, false);
    assert.equal(f.$('start').disabled, true);
    assert.match(f.$('error').textContent, /恢复急停未确认/);
    assert.doesNotMatch(f.$('estopResetFeedback').textContent, /急停已恢复/);
  }
});

test('new emergency stop interrupts recovery and its late response cannot clear the latch', async () => {
  const f = fixture({state: 'estop'}); await f.dashboard.init();
  f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
  const reset = pending(); f.state.handlers['/api/estop/reset'] = () => reset.promise;
  const recovering = f.$('estopReset').onclick();
  await f.$('estopReset').onclick();
  assert.equal(f.posts('/api/estop/reset').length, 1);
  assert.equal(f.$('estop').disabled, false);
  for (const id of ['start', 'load', 'connect', 'download', 'applySettings', 'pause', 'gripperOpen', 'gripperRead', 'gripperEnable']) {
    assert.equal(f.$(id).disabled, true, id);
    await f.$(id).onclick();
  }
  assert.equal(f.posts('/api/pause').length, 0);
  assert.equal(f.posts('/api/gripper').length, 0);
  await f.$('estop').onclick();
  assert.equal(f.posts('/api/estop').length, 1);
  assert.equal(f.$('estopResetConfirmed').checked, false);
  Object.assign(f.state.status, {state: 'paused', estop_latched: false});
  reset.resolve({ok: true, status: 200, json: async () => ({...f.state.status})});
  await recovering;
  assert.equal(f.$('estopNotice').hidden, false);
  assert.equal(f.$('start').disabled, true);
  assert.match(f.$('estopResetFeedback').textContent, /被新的软件急停打断/);
  assert.doesNotMatch(f.$('estopResetFeedback').textContent, /急停已恢复/);
});

test('server reset busy prevents ordinary actions while allowing another emergency stop', async () => {
  const f = fixture({state: 'paused', estop_reset_busy: true}); await f.dashboard.init();
  assert.equal(f.$('estopNotice').hidden, false);
  assert.equal(f.$('estopReset').disabled, true);
  for (const id of ['start', 'connect', 'load', 'gripperOpen', 'pause']) assert.equal(f.$(id).disabled, true, id);
  assert.equal(f.$('estop').disabled, false);
  await f.$('estop').onclick();
  assert.equal(f.posts('/api/estop').length, 1);
});

test('lost successful reset response stays latched until an explicit idempotent retry', async () => {
  const f = fixture({state: 'estop'}); await f.dashboard.init();
  f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
  f.state.handlers['/api/estop/reset'] = async () => {
    Object.assign(f.state.status, {state: 'paused', estop_latched: false, epoch: 3});
    throw new Error('恢复响应连接中断');
  };
  await f.$('estopReset').onclick();
  assert.equal(f.$('estopNotice').hidden, false);
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('estopReset').disabled, false);
  assert.match(f.$('estopResetHint').textContent, /尚未确认恢复/);
  assert.match(f.$('error').textContent, /恢复响应连接中断/);
  await f.dashboard.refreshStatus();
  assert.equal(f.$('start').disabled, true, 'status alone cannot release an uncertain local latch');
  f.state.handlers['/api/estop/reset'] = async () => ({ok: true, status: 200, json: async () => ({...f.state.status})});
  await f.$('estopReset').onclick();
  assert.equal(f.posts('/api/estop/reset').length, 2);
  assert.equal(f.$('estopNotice').hidden, true);
  assert.match(f.$('estopResetFeedback').textContent, /急停已恢复.*保持暂停/);
  assert.equal(f.posts('/api/start').length, 0);
});

test('newer server epoch prevents a stale reset result from clearing local latch', async () => {
  const f = fixture({state: 'estop'}); await f.dashboard.init();
  f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
  f.state.handlers['/api/estop/reset'] = async () => {
    Object.assign(f.state.status, {state: 'paused', estop_latched: false, epoch: 5});
    return {ok: true, status: 200, json: async () => ({state: 'paused', estop_latched: false, estop_reset_busy: false, epoch: 3})};
  };
  await f.$('estopReset').onclick();
  assert.equal(f.$('estopNotice').hidden, false);
  assert.equal(f.$('start').disabled, true);
  assert.match(f.$('error').textContent, /状态未确认或已发生变化/);
});

test('a later server estop replaces the old recovery success message', async () => {
  const f = fixture({state: 'estop'}); await f.dashboard.init();
  f.$('estopResetConfirmed').checked = true; f.$('estopResetConfirmed').onchange();
  await f.$('estopReset').onclick();
  assert.match(f.$('estopResetFeedback').textContent, /急停已恢复/);
  Object.assign(f.state.status, {state: 'estop', estop_latched: true, epoch: 5});
  await f.dashboard.refreshStatus();
  assert.equal(f.$('estopNotice').hidden, false);
  assert.equal(f.$('start').disabled, true);
  assert.match(f.$('estopResetFeedback').textContent, /新的软件急停/);
  assert.doesNotMatch(f.$('estopResetFeedback').textContent, /急停已恢复/);
});


const remoteSha = '9a560012cc277d5b4755cb8e461edb5b238850503a63fb57699567bc9dab823d';
function useRemote(f, overrides = {}) {
  Object.assign(f.state.settings, {inference_device: 'qizhi_h200', sampling_profile: '4416', execution_mode: 'synchronous',
    action_smoothing_alpha: 0.5, max_result_age_s: 0.6, remote_inference: {
      url: 'http://127.0.0.1:8787', checkpoint_sha256: remoteSha, token_env: 'WAM_AUTH_TOKEN',
      timeout_s: 3, max_latency_s: 0.6, image_encoding: 'zlib_rgb', ...overrides,
    }});
  Object.assign(f.state.status, {inference_device: 'qizhi_h200', sampling_profile: '4416', execution_mode: 'synchronous'});
}

test('remote model connects with configured SHA without a local GPU or downloaded checkpoint', async () => {
  const f = fixture({model_ready: false, cuda: {available: false, error: '本机 CUDA 驱动不匹配'}});
  useRemote(f);
  f.state.handlers['/api/checkpoints'] = async () => ({ok: true, status: 200, json: async () => ({directory: '/empty', checkpoints: []})});
  await f.dashboard.init();
  assert.equal(f.$('remoteFields').hidden, false);
  assert.equal(f.$('load').disabled, false);
  assert.equal(f.$('load').textContent, '连接远端模型');
  assert.equal(f.$('download').hidden, true);
  assert.equal(f.$('download').disabled, true);
  assert.equal(f.$('start').disabled, true, 'remote handshake must still finish before control');
  assert.doesNotMatch(f.$('error').textContent, /CUDA/);
  assert.match(f.$('remoteTokenHint').textContent, /WAM_AUTH_TOKEN/);
  await f.$('load').onclick();
  assert.deepEqual(JSON.parse(f.posts('/api/select')[0].body), {id: ''});
  assert.equal(f.posts('/api/connect').length, 0, 'connecting a model never connects hardware');
});

test('remote catalog SHA enables loading without downloading and still rejects missing identities', async () => {
  const f = fixture({model_ready: false, cuda: {available: false}});
  useRemote(f, {checkpoint_sha256: ''});
  let sha = remoteSha;
  f.state.handlers['/api/checkpoints'] = async () => ({ok: true, status: 200, json: async () => ({directory: '/empty',
    checkpoints: [{id: 'step5000', label: 'Step 5000', installed: false, sha256: sha}]})});
  await f.dashboard.init();
  assert.equal(f.$('load').disabled, false);
  await f.$('load').onclick();
  assert.deepEqual(JSON.parse(f.posts('/api/select')[0].body), {id: 'step5000'});
  sha = null; await f.dashboard.refreshCheckpoints();
  assert.equal(f.$('load').disabled, true);
  assert.match(f.$('modelHint').textContent, /完整校验值/);
});

test('remote telemetry displays measured GPU and decomposed latency while hardware and estop stay gated', async () => {
  const f = fixture({cuda: {available: false, error: '本机 CUDA unavailable'}, latency_ms: 291,
    model_diagnostics: {remote: {ready: true, gpu_name: 'NVIDIA H200', device: 'cuda:0', dtype: 'bfloat16',
      inference_backend: 'joint_multirate', checkpoint_name: 'step_005000.pt', checkpoint_sha256: remoteSha,
      sampling_profile: '4416', action_space: 'joint8', optimization: 'compile_cuda_graph'},
    last_inference: {elapsed_ms: 289.6, encode_ms: 6.2, roundtrip_ms: 283.4, transport_overhead_ms: 76.1,
      server_timing_ms: {inference: 213.5, decode: 2, processing: 215.5}}}});
  useRemote(f); await f.dashboard.init();
  assert.equal(f.$('start').disabled, false);
  assert.equal(f.$('cuda').textContent, 'NVIDIA H200');
  assert.equal(f.$('cudaBadge').textContent, '远端已就绪');
  assert.equal(f.$('remoteIdentity').hidden, false);
  assert.match(f.$('remoteModelHash').textContent, new RegExp(remoteSha));
  assert.match(f.$('remoteModelConfig').textContent, /4 \/ 4 \/ 16.*compile_cuda_graph/);
  assert.equal(f.$('latency').textContent, '289.6 ms');
  assert.equal(f.$('serverLatency').textContent, '213.5 ms');
  assert.equal(f.$('transportLatency').textContent, '76.1 ms');
  assert.equal(f.$('encodeLatency').textContent, '6.2 ms');
  assert.equal(f.$('remoteMetrics').hidden, false);
  f.state.status.force_ready = false; await f.dashboard.refreshStatus();
  assert.equal(f.$('start').disabled, true, 'remote GPU does not bypass force readiness');
  f.state.status.force_ready = true;
  f.state.status.error = '远端响应超时'; await f.dashboard.refreshStatus();
  assert.match(f.$('error').textContent, /远端响应超时/);
  await f.$('estop').onclick();
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('load').disabled, true);
  assert.equal(f.$('gripperOpen').disabled, true);
});

test('remote edits preserve transport policy and sync/async switch requires reconnecting the model', async () => {
  const f = fixture(); useRemote(f, {token: 'must-never-be-posted'}); await f.dashboard.init();
  f.choose('remoteUrl', 'http://127.0.0.1:9876');
  f.choose('executionSelect', 'async_relaxed');
  f.choose('smoothingAlpha', '0.35');
  assert.equal(f.$('start').disabled, true);
  await f.$('applySettings').onclick();
  const payload = JSON.parse(f.posts('/api/settings')[0].body);
  assert.equal(payload.inference_device, 'qizhi_h200');
  assert.equal(payload.sampling_profile, '4416');
  assert.equal(payload.execution_mode, 'async_relaxed');
  assert.equal(payload.action_smoothing_alpha, 0.35);
  assert.equal(payload.max_result_age_s, 0.6);
  assert.deepEqual(payload.remote_inference, {url: 'http://127.0.0.1:9876', checkpoint_sha256: remoteSha,
    token_env: 'WAM_AUTH_TOKEN', timeout_s: 3, max_latency_s: 0.6, image_encoding: 'zlib_rgb'});
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('load').disabled, false);
  assert.match(f.$('settingsFeedback').textContent, /重新连接远端模型/);
});

test('device drafts preserve the local smoothing value and cannot silently bypass CUDA', async () => {
  const f = fixture({cuda: {available: false, error: '驱动不可用'}}); await f.dashboard.init();
  assert.equal(f.$('smoothingAlpha').value, '1');
  assert.equal(f.$('start').disabled, true);
  f.choose('inferenceDevice', 'qizhi_h200');
  assert.equal(f.$('smoothingAlpha').value, '0.5');
  assert.equal(f.$('remoteFields').hidden, false);
  assert.equal(f.$('start').disabled, true, 'unapplied remote draft cannot use the existing local model');
  f.choose('smoothingAlpha', '0.3');
  f.choose('inferenceDevice', 'local');
  assert.equal(f.$('smoothingAlpha').value, '1');
  assert.equal(f.$('remoteFields').hidden, true);
  assert.equal(f.$('start').disabled, true);
  assert.match(f.$('error').textContent, /驱动不可用/);
  f.choose('inferenceDevice', 'qizhi_h200');
  assert.equal(f.$('smoothingAlpha').value, '0.3');
});

test('remote address, SHA, and smoothing validation prevent malformed settings posts', async () => {
  for (const [id, value, message] of [
    ['remoteUrl', 'javascript:alert(1)', /HTTP/], ['remoteUrl', 'http://user:secret@localhost:8787', /认证令牌/],
    ['remoteCheckpointSha', '1234', /64 位/], ['smoothingAlpha', '0', /大于 0/],
    ['smoothingAlpha', '1.1', /不超过 1/], ['smoothingAlpha', '', /大于 0/],
  ]) {
    const f = fixture(); useRemote(f); await f.dashboard.init();
    f.choose(id, value); await f.$('applySettings').onclick();
    assert.equal(f.posts('/api/settings').length, 0, `${id}=${value}`);
    assert.match(f.$('error').textContent, message);
  }
});

test('remote controls lock during running inference and a lost local-service connection', async () => {
  const f = fixture({state: 'running', inference_busy: true}); useRemote(f); await f.dashboard.init();
  for (const id of ['inferenceDevice', 'remoteUrl', 'remoteCheckpointSha', 'smoothingAlpha', 'executionSelect']) {
    assert.equal(f.$(id).disabled, true, id);
  }
  await f.$('pause').onclick();
  assert.equal(f.$('inferenceDevice').disabled, true);
  f.state.status.inference_busy = false; await f.dashboard.refreshStatus();
  assert.equal(f.$('inferenceDevice').disabled, false);
  f.state.handlers['/api/status'] = async () => { throw new Error('本机服务断开'); };
  await f.dashboard.refreshStatus();
  assert.equal(f.$('start').disabled, true);
  assert.equal(f.$('load').disabled, true);
  assert.equal(f.$('inferenceDevice').disabled, true);
  assert.equal(f.$('estop').disabled, false);
  assert.equal(f.$('cudaBadge').textContent, '连接中断');
});

test('status polling preserves in-progress remote input and an inference error cannot appear ready', async () => {
  const f = fixture({model_diagnostics: {remote: {ready: true, gpu_name: 'NVIDIA H200'}}});
  useRemote(f); await f.dashboard.init();
  f.$('remoteUrl').value = 'http://127.0.0.1:9'; f.$('remoteUrl').oninput();
  f.$('smoothingAlpha').value = '0.25'; f.$('smoothingAlpha').oninput();
  await f.dashboard.refreshStatus();
  assert.equal(f.$('remoteUrl').value, 'http://127.0.0.1:9');
  assert.equal(f.$('smoothingAlpha').value, '0.25');
  assert.equal(f.$('start').disabled, true);
  Object.assign(f.state.status, {state: 'error', error: '远端请求已超时'});
  await f.dashboard.refreshStatus();
  assert.equal(f.$('cudaBadge').textContent, '远端需检查');
  assert.match(f.$('error').textContent, /远端请求已超时/);
});

test('local settings keep a null result-age limit and device switching supplies the remote bound', async () => {
  const f = fixture();
  Object.assign(f.state.settings, {inference_device: 'local', action_smoothing_alpha: 1, max_result_age_s: null,
    remote_inference: {url: 'http://127.0.0.1:8787', checkpoint_sha256: remoteSha, token_env: 'REALMAN_REMOTE_TOKEN',
      timeout_s: 5, max_latency_s: 1}});
  await f.dashboard.init();
  assert.equal(f.$('applySettings').disabled, true, 'null does not become an unapplied zero');
  f.choose('samplingProfile', '4416'); await f.$('applySettings').onclick();
  assert.equal(JSON.parse(f.posts('/api/settings')[0].body).max_result_age_s, null);
  f.choose('inferenceDevice', 'qizhi_h200'); await f.$('applySettings').onclick();
  assert.equal(JSON.parse(f.posts('/api/settings')[1].body).max_result_age_s, 0.6);
  f.choose('inferenceDevice', 'local'); await f.$('applySettings').onclick();
  assert.equal(JSON.parse(f.posts('/api/settings')[2].body).max_result_age_s, null);
  assert.equal(f.$('smoothingAlpha').value, '1');
});

test('remote optimization display prefers the effective mode and includes the captured scope', async () => {
  const f = fixture({model_diagnostics: {remote: {ready: true, sampling_profile: '4416',
    optimization: {requested_mode: 'compile_cuda_graph', effective_mode: 'eager', graph_scope: 'sampler'}}}});
  useRemote(f); await f.dashboard.init();
  assert.match(f.$('remoteModelConfig').textContent, /eager.*图捕获 sampler/);
  assert.doesNotMatch(f.$('remoteModelConfig').textContent, /compile_cuda_graph/);
});
