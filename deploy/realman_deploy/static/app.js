/* Browser controller; the factory is exported for hardware-free Node tests. */
(function (root) {
  'use strict';
  function createDashboard({document, fetch, setTimeout, clearTimeout, setInterval, clearInterval, now = () => new Date()}) {
    const $ = id => document.getElementById(id);
    const phases = {inferring: '停驻推理中', waiting_for_observation: '停驻取新观测',
      executing_and_inferring: '执行并推理', executing: '执行中', waiting: '等待新动作',
      idle: '空闲', ready: '就绪', paused: '已暂停', loading: '模型加载中', stopping: '停止中',
      stopped: '已停止', error: '错误', estop: '急停锁定', running: '运行中'};
    let status = null, online = false, settings = null, checkpoints = [], busy = false;
    let pausePending = false, estopPending = false, localEstop = false, interval = null;
    let actionError = '', stopError = '', statusError = '', settingsError = '', catalogError = '';
    let statusPromise = null, catalogPromise = null, settingsPromise = null;
    let gripperDebug = null, gripperBusy = false, gripperError = '', gripperFeedback = '';
    let gripperPromise = null;
    let draft = {sampling_profile: '161616', execution_mode: 'synchronous'};
    const gripperActions = {gripperConnect: 'connect', gripperRead: 'status', gripperEnable: 'enable',
      gripperDisable: 'disable', gripperClear: 'clear_error', gripperOpen: 'open', gripperClose: 'close', gripperMove: 'move'};
    const gripperLabels = {connect: '连接夹爪', status: '读取状态', enable: '使能', disable: '失能',
      clear_error: '清除报警', open: '打开', close: '闭合', move: '移动到目标开度'};
    const text = (id, value) => { $(id).textContent = value; };
    function badge(id, label, kind = '') { text(id, label); $(id).className = `badge ${kind}`; }
    const inactive = () => online && status && !status.inference_busy &&
      ['idle', 'ready', 'paused', 'error'].includes(status.state) && !localEstop;
    const gripperPending = () => gripperBusy || Boolean(status?.gripper_debug_busy) || Boolean(gripperDebug?.busy);
    const occupied = () => busy || Boolean(status?.hardware_connecting) || gripperPending();
    const changed = () => settings && (draft.sampling_profile !== settings.sampling_profile ||
      draft.execution_mode !== settings.execution_mode);
    const milliseconds = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${Number(value).toFixed(1)} ms`;
    function showErrors() {
      const message = [stopError, actionError, statusError, settingsError, catalogError,
        status?.error, status?.hardware?.error, status?.cuda?.error].filter(Boolean).filter((x, i, a) => a.indexOf(x) === i).join('\n');
      text('error', message); $('error').hidden = !message;
    }
    async function api(path, body, timeoutMs = body === undefined ? 5000 : 60000) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(path, {signal: controller.signal,
          method: body === undefined ? 'GET' : 'POST', headers: {'Content-Type': 'application/json'},
          body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store'});
        let value;
        try { value = await response.json(); }
        catch (_) { throw new Error(`服务返回无法解析的响应（HTTP ${response.status}）：${path}`); }
        if (!response.ok) throw new Error(value.error || `HTTP ${response.status}：${path}`);
        return value;
      } catch (error) {
        if (error.name === 'AbortError') throw new Error(`请求超时：${path}。服务端操作可能仍在进行，请先刷新状态。`);
        throw error;
      } finally { clearTimeout(timer); }
    }
    function renderSettings() {
      $('samplingProfile').value = draft.sampling_profile;
      $('executionSelect').value = draft.execution_mode;
      $('executionField').hidden = draft.sampling_profile !== '4416' || settings?.supported === false;
      text('profileHint', draft.sampling_profile === '4416'
        ? 'Video / Force 各采样 4 步，Action 采样 16 步；可选同步或异步执行。实际延迟随设备与输入变化。'
        : 'Video / Force / Action 各采样 16 步，用于基准对照。');
      const locked = !settings || settings.supported === false || occupied() || !inactive() || pausePending || estopPending;
      $('samplingProfile').disabled = locked;
      $('executionSelect').disabled = locked;
      $('applySettings').disabled = locked || !changed();
      badge('settingsState', !settings ? '未读取' : settings.supported === false ? '不适用' : changed() ? '尚未应用' : '已应用', changed() ? 'warn' : settings?.supported === false ? '' : settings ? 'good' : '');
      text('settingsHint', settings?.supported === false ? '当前模型使用旧版动作空间，4 / 4 / 16 采样仅适用于 joint8 模型。' : status?.state === 'running' ? '先暂停运行，等待当前推理完成后修改。'
        : status?.inference_busy ? '正在等待当前推理结束，设置暂不可修改。'
        : '应用新设置后需要重新加载模型；设置不会修改 checkpoint。');
    }
    function renderButtons() {
      const selected = checkpoints.find(item => item.id === $('checkpoint').value);
      const locked = occupied() || !inactive() || pausePending || estopPending;
      $('checkpoint').disabled = occupied() || !checkpoints.length || !inactive();
      $('download').disabled = occupied() || !selected || selected.installed || selected.source === 'local' || localEstop;
      $('load').disabled = locked || !selected?.installed || !status?.cuda?.available || !settings || changed();
      $('connect').disabled = locked;
      $('start').disabled = locked || !settings || !status?.model_ready || !status?.robot_ready || !status?.cameras_ready ||
        !status?.force_ready || !status?.gripper_ready || !status?.cuda?.available || changed() ||
        !['ready', 'paused', 'idle'].includes(status?.state);
      // Stop controls have their own lane: loading/downloading must not block them.
      $('pause').disabled = pausePending || estopPending || localEstop || status?.state === 'estop';
      $('pause').textContent = pausePending ? '正在暂停…' : status?.state === 'error' ? '复位错误 / 保持停止' : '暂停 / 缓停';
      $('estop').disabled = estopPending || status?.state === 'estop';
      $('estop').textContent = estopPending ? '正在急停…' : '软件急停';
      $('refresh').disabled = Boolean(catalogPromise) || occupied();
      renderSettings(); renderGripper();
    }
    function renderGripper() {
      const g = gripperDebug;
      const supported = g?.supported === true, connected = g?.connected === true;
      const pending = gripperPending();
      const locked = occupied() || !inactive() || pausePending || estopPending || !supported;
      const alarmClear = Number.isFinite(g?.alarm) && g.alarm === 0;
      const motionLocked = locked || !connected || g?.enabled !== true || !alarmClear || g?.moving === true;
      badge('gripperDebugBadge', !online ? '服务离线' : !g ? '未读取' : !supported ? '不支持' : pending ? '操作中'
        : g.error || g.command_error || g.last_command_error || (connected && !alarmClear) ? '需检查' : connected ? '已连接' : '未连接',
        g?.error || g?.command_error || g?.last_command_error || (connected && !alarmClear) ? 'bad' : connected && !pending ? 'good' : 'warn');
      text('gripperConnection', connected ? '已连接' : g ? '未连接' : '—');
      text('gripperEnabled', !connected || typeof g?.enabled !== 'boolean' ? '—' : g.enabled ? '已使能' : '已失能');
      text('gripperOpening', connected && Number.isFinite(g?.opening) ? `${(g.opening * 100).toFixed(1)}%` : '—');
      text('gripperPosition', connected && Number.isFinite(g?.position) ? String(g.position) : '—');
      text('gripperMotion', pending ? '操作处理中' : !connected ? '—' : g.moving === true ? '运动中' : g.moving === false ? '静止' : '待确认');
      text('gripperAlarm', connected && Number.isFinite(g?.alarm) ? `${g.alarm_hex || `0x${g.alarm.toString(16).toUpperCase().padStart(4, '0')}`} (${g.alarm})` : '—');
      $('gripperAlarm').className = connected && !alarmClear ? 'bad' : '';
      text('gripperAlarmLabels', !connected ? '连接后读取报警反馈' : Array.isArray(g.alarm_labels) && g.alarm_labels.length
        ? g.alarm_labels.join('；') : alarmClear ? '无报警' : '报警状态待确认');
      $('gripperConnect').disabled = locked;
      if (!gripperBusy) text('gripperConnect', connected ? '重新连接夹爪' : '连接夹爪');
      for (const id of ['gripperRead', 'gripperDisable', 'gripperClear']) $(id).disabled = locked || !connected;
      $('gripperEnable').disabled = locked || !connected || !alarmClear || g?.enabled === true;
      for (const id of ['gripperOpen', 'gripperClose', 'gripperMove', 'gripperTarget', 'gripperSpeed', 'gripperTorque']) $(id).disabled = motionLocked;
      text('gripperGuard', !online ? '服务连接中断，调试暂不可用。'
        : !supported ? '等待服务提供夹爪调试状态。'
        : localEstop || status?.state === 'estop' ? '软件急停已锁定，夹爪调试不可用。'
        : !inactive() || pausePending || estopPending ? '请先暂停推理，等待加载、停止或当前推理完成。'
        : pending ? '正在等待夹爪反馈；暂停和软件急停仍可使用。'
        : busy || status?.hardware_connecting ? '其他操作或硬件连接进行中，稍后可调试夹爪。'
        : !connected ? '先连接夹爪。此连接无需模型、GPU 或双相机就绪。'
        : !alarmClear ? '请检查报警并清除；清除后保持失能，再手动使能。'
        : g.enabled !== true ? '夹爪已连接；手动使能后可执行开合。'
        : g.moving ? '夹爪仍在运动，请等待反馈或暂停。'
        : '0% 为闭合，100% 为完全打开；命令完成以硬件反馈为准。');
      const operation = typeof g?.operation === 'string' ? (gripperLabels[g.operation] || g.operation)
        : g?.operation?.message || g?.operation?.action && (gripperLabels[g.operation.action] || g.operation.action);
      text('gripperFeedback', gripperFeedback || (operation ? `最近操作：${operation}` : ''));
      const error = [gripperError, g?.error, g?.command_error, g?.last_command_error].filter(Boolean).filter((x, i, a) => a.indexOf(x) === i).join('\n');
      text('gripperError', error); $('gripperError').hidden = !error;
    }
    function renderStatus() {
      const s = status;
      if (!s) { renderButtons(); showErrors(); return; }
      badge('badge', online ? (phases[s.state] || s.state) : '服务连接中断', !online || ['error', 'estop'].includes(s.state) ? 'bad' : s.state === 'running' ? 'good' : 'warn');
      badge('robot', s.robot_ready ? '就绪' : '未就绪', s.robot_ready ? 'good' : 'warn');
      badge('cameras', s.cameras_ready ? '双路就绪' : '未就绪', s.cameras_ready ? 'good' : 'warn');
      text('force', s.force_ready ? '就绪' : '未就绪');
      text('gripper', s.gripper_ready ? '就绪' : '未就绪');
      badge('cudaBadge', s.cuda?.available ? '可用' : s.cuda?.checking ? '检查中' : '不可用', s.cuda?.available ? 'good' : s.cuda?.checking ? 'warn' : 'bad');
      text('cuda', s.cuda?.available ? `${s.cuda.device}${s.cuda.total_bytes ? ` · ${(s.cuda.total_bytes / 1e9).toFixed(1)} GB` : ''}` : s.cuda?.checking ? '正在检查 CUDA 驱动' : '请检查 CUDA 环境与驱动');
      const diagnostic = s.model_diagnostics || {};
      text('cudaDetail', [diagnostic.device, diagnostic.dtype, diagnostic.inference_backend].filter(Boolean).join(' · ') || '精度与计算后端在加载时确定');
      badge('modelBadge', s.model_ready ? '模型已加载' : s.state === 'loading' ? '加载中' : '未加载', s.model_ready ? 'good' : 'warn');
      text('modelMode', `双视角观测 · ${s.model_ready ? (s.reference_mode === 'none' ? '无参考轨迹' : '滚动 AE 参考') : '模型未加载'} · 30 Hz 动作接口`);
      const profile = s.sampling_profile || settings?.sampling_profile;
      badge('activeProfile', settings?.supported === false ? '旧版采样配置' : profile === '4416' ? '4 / 4 / 16 · 平衡' : profile === '161616' ? '16 / 16 / 16 · 基准' : '采样设置待确认');
      text('executionMode', s.execution_mode === 'synchronous' ? '同步推理' : s.execution_mode === 'async_relaxed' ? '异步推理' : '异步严格时序');
      text('executionPhase', phases[s.execution_phase] || phases[s.state] || s.state);
      text('inferencePhase', s.inference_busy ? '模型推理中' : '模型空闲');
      text('latency', milliseconds(s.latency_ms));
      text('inputAge', milliseconds(s.input_age_at_result_ms));
      text('queue', String(s.action_queue ?? 0)); text('discarded', String(s.discarded_results ?? 0));
      text('state', `${s.state} / ${s.epoch ?? 0}`);
      text('sequence', s.observation_sequence == null || s.observation_sequence < 0 ? '—' : String(s.observation_sequence));
      text('modeHint', s.execution_mode === 'synchronous'
        ? '同步：完整执行当前动作块后停驻，获取新观测并推理，再执行下一块。'
        : s.execution_mode === 'async_relaxed'
          ? '异步：执行当前动作块时计算下一块；动作缓冲耗尽时缓停等待。模型使用本轮推理开始时的观测。'
          : '异步严格时序：按观测时间丢弃过期动作，动作缓冲耗尽时停止。');
      if (s.execution_mode === 'async_relaxed' && s.latency_ms > 32 / 30 * 1000) {
        text('modeHint', $('modeHint').textContent + ' 当前模型延迟超过一块动作约 1.07 秒的时长，可能出现等待间隔。');
      }
      text('diagnostics', JSON.stringify({model: diagnostic, cuda: s.cuda, hardware: s.hardware, gripper: gripperDebug}, null, 2));
      $('estopNotice').hidden = !(localEstop || s.state === 'estop');
      renderButtons(); showErrors();
    }
    async function refreshStatus(fresh = false) {
      if (statusPromise) { if (!fresh) return statusPromise; await statusPromise; }
      const task = (async () => {
        try {
          const value = await api('/api/status');
          if (!value || typeof value.state !== 'string') throw new Error('服务状态响应缺少运行状态');
          status = value; online = true; statusError = '';
          if (value.gripper_debug && typeof value.gripper_debug.supported === 'boolean') gripperDebug = value.gripper_debug;
          if (status.state === 'estop') localEstop = true;
          text('updated', `更新于 ${now().toLocaleTimeString('zh-CN', {hour12: false})}`);
        } catch (error) {
          online = false; statusError = error.message;
          badge('badge', '服务连接中断', 'bad');
        } finally { renderStatus(); }
      })();
      statusPromise = task;
      try { await task; } finally { if (statusPromise === task) statusPromise = null; }
    }
    async function refreshCheckpoints() {
      if (catalogPromise) return catalogPromise;
      const task = (async () => {
        try {
          const value = await api('/api/checkpoints');
          if (!Array.isArray(value.checkpoints)) throw new Error('checkpoint 列表格式错误');
          checkpoints = value.checkpoints;
          const select = $('checkpoint'), previous = select.value;
          select.replaceChildren();
          for (const checkpoint of checkpoints) {
            const option = document.createElement('option'); option.value = checkpoint.id;
            option.textContent = `${checkpoint.label || checkpoint.id} · ${checkpoint.installed ? '本地文件' : '待下载'}`;
            select.append(option);
          }
          if (checkpoints.some(item => item.id === previous)) select.value = previous;
          else if (checkpoints.length) select.value = checkpoints[0].id;
          text('modelHint', checkpoints.length ? `本地目录：${value.directory} · ${checkpoints.length} 个模型`
            : `本地目录：${value.directory}。放入 checkpoint，或配置带 SHA256 校验的下载来源。`);
          catalogError = '';
        } catch (error) { catalogError = error.message; }
      })();
      catalogPromise = task; renderButtons();
      try { await task; } finally { catalogPromise = null; renderButtons(); showErrors(); }
    }
    async function refreshSettings() {
      if (settingsPromise) return settingsPromise;
      const task = (async () => {
        try {
          const value = await api('/api/settings');
          if (!['161616', '4416'].includes(value.sampling_profile)) throw new Error('采样设置响应无效');
          const preserveDraft = changed();
          settings = value;
          if (!preserveDraft) draft = {sampling_profile: value.sampling_profile, execution_mode: value.execution_mode};
          settingsError = '';
        } catch (error) { settingsError = error.message; }
      })();
      settingsPromise = task;
      try { await task; } finally { settingsPromise = null; renderStatus(); }
    }
    async function action(id, path, body = {}, after) {
      if (occupied() || $(id).disabled) return;
      busy = true; actionError = ''; const label = $(id).textContent;
      text(id, {connect: '正在连接…', load: '正在加载…', download: '正在下载…', applySettings: '正在应用…'}[id] || '处理中…');
      renderButtons(); showErrors();
      try {
        const result = await api(path, body);
        if (after) await after(result);
      } catch (error) { actionError = error.message; }
      finally { busy = false; text(id, label); await refreshStatus(true); renderButtons(); showErrors(); }
    }
    async function refreshGripper() {
      if (gripperPromise) return gripperPromise;
      const task = (async () => {
        try {
          const value = await api('/api/gripper');
          if (!value || typeof value.supported !== 'boolean') throw new Error('夹爪调试状态响应无效');
          gripperDebug = value;
        } catch (error) { gripperError = error.message; }
        finally { renderButtons(); }
      })();
      gripperPromise = task;
      try { await task; } finally { if (gripperPromise === task) gripperPromise = null; }
    }
    async function gripperAction(id) {
      if (occupied() || $(id).disabled) return;
      const command = gripperActions[id], body = {action: command};
      const moves = ['open', 'close', 'move'].includes(command);
      if (moves) {
        const speed = Number($('gripperSpeed').value), torque = Number($('gripperTorque').value);
        const target = Number($('gripperTarget').value);
        if (!Number.isInteger(speed) || speed < 1 || speed > 100 || !Number.isInteger(torque) || torque < 1 || torque > 100) {
          gripperError = '速度和力度必须为 1–100 的整数。'; renderGripper(); return;
        }
        if (command === 'move' && (!$('gripperTarget').value.trim() || !Number.isFinite(target) || target < 0 || target > 100)) {
          gripperError = '目标开度必须为 0–100%。'; renderGripper(); return;
        }
        body.speed = speed; body.torque = torque;
        if (command === 'move') body.opening = target / 100;
      }
      gripperBusy = true; gripperError = '';
      gripperFeedback = `${gripperLabels[command]}：正在等待硬件反馈…`;
      const label = $(id).textContent; text(id, '处理中…'); renderButtons();
      try {
        const result = await api('/api/gripper', body, 20000);
        if (!result || typeof result.connected !== 'boolean' || result.supported !== true) throw new Error('夹爪操作响应缺少有效反馈，请读取状态确认。');
        gripperDebug = result;
        const detail = result.operation_result;
        const interrupted = result.command_completed === false || detail?.cancelled || detail?.canceled || detail?.stopped ||
          detail?.stop_confirmed === false || detail?.error || ['cancelled', 'canceled', 'stopped', 'failed', 'error'].includes(detail?.status) ||
          typeof detail === 'string' && /取消|中断|停止|cancel|interrupt|stop|failed|error/i.test(detail);
        if (interrupted) throw new Error(result.command_error || result.last_command_error || detail?.error || detail?.message ||
          (typeof detail === 'string' ? detail : '夹爪命令未完成或已被停止，请读取状态确认。'));
        const validFeedback = result.connected && (command === 'enable' ? result.ready === true && result.enabled === true && result.alarm === 0 && !result.error
          : command === 'disable' ? result.enabled === false
          : command === 'clear_error' ? result.enabled === false && result.alarm === 0
          : moves ? result.ready === true && result.enabled === true && result.alarm === 0 && !result.error
          : true);
        if (!validFeedback) {
          throw new Error(result.error || `${gripperLabels[command]}未确认，请读取状态检查反馈。`);
        } else if (result.busy || (moves && result.moving)) {
          gripperFeedback = `${gripperLabels[command]}尚未完成，请等待硬件反馈。`;
        } else {
          gripperFeedback = command === 'clear_error' ? '报警已清除，夹爪保持失能（反馈已确认）。'
            : command === 'status' ? '已读取夹爪状态。' : `${gripperLabels[command]}完成（反馈已确认）。`;
        }
      } catch (error) { gripperError = error.message; gripperFeedback = `${gripperLabels[command]}未确认。`; }
      finally {
        gripperBusy = false; text(id, label);
        await refreshStatus(true); renderButtons(); showErrors();
      }
    }
    async function stop(emergency) {
      if (emergency ? estopPending || status?.state === 'estop' : pausePending || localEstop) return;
      if (emergency) { estopPending = true; localEstop = true; $('estopNotice').hidden = false; }
      else pausePending = true;
      actionError = ''; renderButtons(); showErrors();
      try { await api(emergency ? '/api/estop' : '/api/pause', {}, 10000); stopError = ''; }
      catch (error) { stopError = `${emergency ? '急停' : '暂停'}未确认：${error.message}${emergency ? ' 请使用控制柜物理急停。' : ''}`; }
      finally {
        if (emergency) estopPending = false; else pausePending = false;
        await refreshStatus(true); renderButtons(); showErrors();
      }
    }
    $('checkpoint').onchange = renderButtons;
    $('samplingProfile').onchange = () => {
      draft.sampling_profile = $('samplingProfile').value;
      if (draft.sampling_profile !== '4416') draft.execution_mode = 'synchronous';
      text('settingsFeedback', ''); renderButtons();
    };
    $('executionSelect').onchange = () => {
      draft.execution_mode = draft.sampling_profile === '4416' ? $('executionSelect').value : 'synchronous';
      text('settingsFeedback', ''); renderButtons();
    };
    $('applySettings').onclick = () => action('applySettings', '/api/settings', {...draft}, async () => {
      // Refresh the persisted settings before enabling the next model load.
      settings = null; await refreshSettings();
      text('settingsFeedback', settings ? '已应用，请重新加载模型。' : '请刷新确认设置。');
    });
    $('refresh').onclick = refreshCheckpoints;
    $('reload').onclick = () => Promise.all([refreshStatus(true), refreshSettings()]);
    $('download').onclick = () => action('download', '/api/download', {id: $('checkpoint').value}, async result => {
      text('downloadProgress', `下载完成：${result.path}`); await refreshCheckpoints();
    });
    $('load').onclick = () => action('load', '/api/select', {id: $('checkpoint').value});
    $('connect').onclick = () => action('connect', '/api/connect');
    $('start').onclick = () => action('start', '/api/start');
    $('pause').onclick = () => stop(false);
    $('estop').onclick = () => stop(true);
    for (const id of Object.keys(gripperActions)) $(id).onclick = () => gripperAction(id);
    async function init() {
      renderButtons();
      await Promise.all([refreshCheckpoints(), refreshSettings(), refreshStatus(), refreshGripper()]);
      interval = setInterval(() => { void refreshStatus(); }, 1000);
    }
    return {init, refreshStatus, refreshSettings, refreshCheckpoints, dispose: () => clearInterval(interval)};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {createDashboard};
  else root.RealmanDashboard = createDashboard({document: root.document, fetch: root.fetch.bind(root),
    setTimeout: root.setTimeout.bind(root), clearTimeout: root.clearTimeout.bind(root),
    setInterval: root.setInterval.bind(root), clearInterval: root.clearInterval.bind(root)});
  if (root.RealmanDashboard) void root.RealmanDashboard.init();
})(typeof window !== 'undefined' ? window : globalThis);
