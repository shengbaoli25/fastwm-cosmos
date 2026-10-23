'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createDashboard} = require('../../realman_deploy/static/app.js');

const html = fs.readFileSync(path.join(__dirname, '../../realman_deploy/static/index.html'), 'utf8');
function fixture(overrides = {}) {
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map((match) => [match[1], {
    textContent: '', value: '', disabled: false, hidden: false, className: '', children: [],
    replaceChildren() { this.children = []; this.value = ''; }, append(child) { this.children.push(child); },
  }]));
  const state = {status: {state: 'ready', epoch: 1, model_ready: true, robot_ready: true,
    cameras_ready: true, force_ready: true, gripper_ready: true, inference_busy: false,
    sampling_profile: '161616', execution_mode: 'synchronous', execution_phase: 'ready',
    cuda: {available: true, device: 'Test GPU', total_bytes: 24000000000}, ...overrides},
    settings: {sampling_profile: '161616', execution_mode: 'synchronous', supported: true}, calls: [], handlers: {}, timers: new Map(), nextTimer: 0};
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
    else if (url === '/api/estop') { state.status.state = 'estop'; result = {state: 'estop'}; }
    else if (url === '/api/pause') { state.status.state = 'paused'; result = {state: 'paused'}; }
    else result = {};
    return {ok: true, status: 200, json: async () => result};
  };
  const dashboard = createDashboard({document: {getElementById: id => { assert(elements.has(id), `Missing #${id}`); return elements.get(id); },
    createElement: () => ({textContent: '', value: ''})}, fetch,
    setTimeout: callback => { state.timers.set(++state.nextTimer, callback); return state.nextTimer; },
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
