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
    let catalogDirectory = '读取中';
    let pausePending = false, estopPending = false, localEstop = false, interval = null;
    let estopResetPending = false, estopGeneration = 0, estopResetError = '', estopResetFeedback = '';
    let actionError = '', stopError = '', statusError = '', settingsError = '', catalogError = '';
    let statusPromise = null, catalogPromise = null, settingsPromise = null;
    let gripperDebug = null, gripperBusy = false, gripperError = '', gripperFeedback = '';
    let gripperPromise = null;
    let draft = {sampling_profile: '161616', execution_mode: 'synchronous', inference_device: 'local',
      remote_inference: {url: '', checkpoint_sha256: '', token_env: 'REALMAN_REMOTE_TOKEN', timeout_s: 5, max_latency_s: 1},
      action_smoothing_alpha: 1, max_result_age_s: null};
    let deviceTouched = false, remoteTouched = false, smoothingTouched = false;
    const smoothingByDevice = {local: 1, qizhi_h200: 0.5};
    const resultAgeByDevice = {local: null, qizhi_h200: 0.6};
    const gripperActions = {gripperConnect: 'connect', gripperRead: 'status', gripperEnable: 'enable',
      gripperDisable: 'disable', gripperClear: 'clear_error', gripperOpen: 'open', gripperClose: 'close', gripperMove: 'move'};
    const gripperLabels = {connect: '连接夹爪', status: '读取状态', enable: '使能', disable: '失能',
      clear_error: '清除报警', open: '打开', close: '闭合', move: '移动到目标开度'};
    const text = (id, value) => { $(id).textContent = value; };
    function badge(id, label, kind = '') { text(id, label); $(id).className = `badge ${kind}`; }
    const serverEstop = () => status?.estop_latched === true || status?.state === 'estop';
    const resettingEstop = () => estopResetPending || status?.estop_reset_busy === true;
    const inactive = () => online && status && !status.inference_busy &&
      ['idle', 'ready', 'paused', 'error'].includes(status.state) && !localEstop && !serverEstop();
    const gripperPending = () => gripperBusy || Boolean(status?.gripper_debug_busy) || Boolean(gripperDebug?.busy);
    const occupied = () => busy || Boolean(status?.hardware_connecting) || gripperPending() || resettingEstop();
    const remoteMode = () => (draft.inference_device || settings?.inference_device || status?.inference_device) === 'qizhi_h200';
    const remoteConfig = value => {
      const {token: ignoredToken, ...source} = value?.remote_inference || {};
      return {...source, url: source.url || '', checkpoint_sha256: source.checkpoint_sha256 || '',
        token_env: source.token_env || 'REALMAN_REMOTE_TOKEN', timeout_s: source.timeout_s ?? 5,
        max_latency_s: source.max_latency_s ?? 1};
    };
    const normalisedSettings = value => ({
      sampling_profile: value?.sampling_profile || '161616', execution_mode: value?.execution_mode || 'synchronous',
      inference_device: value?.inference_device || 'local', remote_inference: remoteConfig(value),
      action_smoothing_alpha: Number.isFinite(Number(value?.action_smoothing_alpha)) ? Number(value.action_smoothing_alpha) : 1,
      max_result_age_s: value?.max_result_age_s == null
        ? value?.inference_device === 'qizhi_h200' ? 0.6 : null
        : Number.isFinite(Number(value.max_result_age_s)) ? Number(value.max_result_age_s) : null,
    });
    const changed = () => {
      if (!settings) return false;
      const base = normalisedSettings(settings), remote = remoteConfig(draft);
      const configuredRemote = remoteConfig(base);
      return draft.sampling_profile !== base.sampling_profile || draft.execution_mode !== base.execution_mode ||
        draft.inference_device !== base.inference_device || Number(draft.action_smoothing_alpha) !== base.action_smoothing_alpha ||
        draft.max_result_age_s !== base.max_result_age_s ||
        remote.url !== configuredRemote.url || remote.checkpoint_sha256 !== configuredRemote.checkpoint_sha256 ||
        remote.token_env !== configuredRemote.token_env || Number(remote.timeout_s) !== Number(configuredRemote.timeout_s) ||
        Number(remote.max_latency_s) !== Number(configuredRemote.max_latency_s);
    };
    const settingsPayload = () => {
      const payload = {sampling_profile: draft.sampling_profile, execution_mode: draft.execution_mode};
      const hasDeviceFields = settings && ('inference_device' in settings || 'remote_inference' in settings ||
        'action_smoothing_alpha' in settings || 'max_result_age_s' in settings);
      if (hasDeviceFields || deviceTouched || draft.inference_device !== 'local') payload.inference_device = draft.inference_device;
      if (hasDeviceFields || remoteTouched || draft.inference_device !== 'local') payload.remote_inference = {...remoteConfig(draft)};
      if (hasDeviceFields || smoothingTouched || Number(draft.action_smoothing_alpha) !== 1) payload.action_smoothing_alpha = Number(draft.action_smoothing_alpha);
      if (hasDeviceFields || deviceTouched || draft.max_result_age_s != null) payload.max_result_age_s = draft.max_result_age_s;
      return payload;
    };
    const milliseconds = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${Number(value).toFixed(1)} ms`;
    function showErrors() {
      const message = [stopError, estopResetError, actionError, statusError, settingsError, catalogError,
        status?.error, status?.hardware?.error, remoteMode() ? null : status?.cuda?.error].filter(Boolean).filter((x, i, a) => a.indexOf(x) === i).join('\n');
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
      const remote = remoteMode(), config = remoteConfig(draft);
      $('inferenceDevice').value = draft.inference_device;
      $('remoteFields').hidden = !remote;
      $('remoteUrl').value = config.url;
      $('remoteCheckpointSha').value = config.checkpoint_sha256;
      $('smoothingAlpha').value = String(draft.action_smoothing_alpha);
      text('remoteTokenHint', `认证令牌通过本机服务启动环境 ${config.token_env} 提供。远端权重无需下载到本机。`);
      $('samplingProfile').value = draft.sampling_profile;
      $('executionSelect').value = draft.execution_mode;
      $('executionField').hidden = draft.sampling_profile !== '4416' || settings?.supported === false;
      text('profileHint', draft.sampling_profile === '4416'
        ? 'Video / Force 各采样 4 步，Action 采样 16 步；可选同步或异步执行。实际延迟随设备与输入变化。'
        : 'Video / Force / Action 各采样 16 步，用于基准对照。');
      const locked = !settings || settings.supported === false || occupied() || !inactive() || pausePending || estopPending;
      $('samplingProfile').disabled = locked;
      $('executionSelect').disabled = locked;
      for (const id of ['inferenceDevice', 'remoteUrl', 'remoteCheckpointSha', 'smoothingAlpha']) $(id).disabled = locked;
      $('applySettings').disabled = locked || !changed();
      badge('settingsState', !settings ? '未读取' : settings.supported === false ? '不适用' : changed() ? '尚未应用' : '已应用', changed() ? 'warn' : settings?.supported === false ? '' : settings ? 'good' : '');
      text('settingsHint', settings?.supported === false ? '当前模型使用旧版动作空间，4 / 4 / 16 采样仅适用于 joint8 模型。' : status?.state === 'running' ? '先暂停运行，等待当前推理完成后修改。'
        : status?.inference_busy ? '正在等待当前推理结束，设置暂不可修改。'
        : '应用新设置后需要重新加载模型；设置不会修改 checkpoint。');
    }
    function renderButtons() {
      const selected = checkpoints.find(item => item.id === $('checkpoint').value);
      const remote = remoteMode(), config = remoteConfig(draft);
      const expectedSha = config.checkpoint_sha256.trim() || selected?.sha256 || '';
      const remoteConfigured = Boolean(config.url.trim()) && /^[a-fA-F0-9]{64}$/.test(expectedSha);
      const locked = occupied() || !inactive() || pausePending || estopPending;
      $('checkpoint').disabled = occupied() || !checkpoints.length || !inactive();
      $('download').hidden = remote;
      $('download').disabled = remote || occupied() || !selected || selected.installed || selected.source === 'local' || localEstop;
      $('downloadProgress').hidden = remote;
      $('load').disabled = locked || (remote ? !remoteConfigured : !selected?.installed || !status?.cuda?.available) || !settings || changed();
      if (!busy) text('load', remote ? '连接远端模型' : '加载模型');
      text('modelDescription', remote ? '校验远端模型身份，动作仍在本机执行' : '选择权重并加载到当前 CUDA 设备');
      if (remote) text('modelHint', config.checkpoint_sha256.trim()
        ? '按填写的 SHA256 校验远端权重；可不选择本地模型。'
        : selected?.sha256 ? '按所选模型的 SHA256 校验远端权重，无需下载本地文件。'
        : '选择带 SHA256 的模型，或填写远端权重的完整校验值后应用设置。');
      else text('modelHint', checkpoints.length ? `本地目录：${catalogDirectory} · ${checkpoints.length} 个模型`
        : `本地目录：${catalogDirectory}。放入 checkpoint，或配置带 SHA256 校验的下载来源。`);
      $('connect').disabled = locked;
      $('start').disabled = locked || !settings || !status?.model_ready || !status?.robot_ready || !status?.cameras_ready ||
        !status?.force_ready || !status?.gripper_ready || (!remote && !status?.cuda?.available) || changed() ||
        !['ready', 'paused', 'idle'].includes(status?.state);
      // Stop controls have their own lane: loading/downloading must not block them.
      $('pause').disabled = pausePending || estopPending || localEstop || serverEstop() || resettingEstop();
      $('pause').textContent = pausePending ? '正在暂停…' : status?.state === 'error' ? '复位错误 / 保持停止' : '暂停 / 缓停';
      $('estop').disabled = estopPending || (serverEstop() && !resettingEstop());
      $('estop').textContent = estopPending ? '正在急停…' : '软件急停';
      $('refresh').disabled = Boolean(catalogPromise) || occupied();
      renderEstopRecovery(); renderSettings(); renderGripper();
    }
    function renderEstopRecovery() {
      const latched = localEstop || serverEstop(), pending = resettingEstop();
      const supported = status?.estop_reset_supported === true;
      const retryConfirmation = localEstop && status?.state === 'paused' && status?.estop_latched === false;
      const available = online && supported && (serverEstop() || retryConfirmation) && !occupied() && !pausePending && !estopPending && !status?.inference_busy;
      $('estopNotice').hidden = !latched && !pending;
      $('estopResetConfirmed').disabled = !available;
      $('estopReset').disabled = !available || !$('estopResetConfirmed').checked;
      text('estopReset', pending ? '正在恢复…' : '恢复急停');
      text('estopResetHint', !online ? '服务连接中断，暂不能确认或恢复急停。网络停止未确认时，请使用控制柜物理急停。'
        : !supported ? '当前服务尚未提供急停恢复接口，请先更新服务。急停锁定保持生效。'
        : pending ? '正在确认停止状态并恢复软件急停。恢复仅进入暂停，不会自动使能或开始推理；软件急停仍可打断恢复。'
        : retryConfirmation ? '服务显示已暂停，但本页尚未确认恢复。检查现场后重试恢复确认；不会自动开始推理。'
        : !serverEstop() ? '急停尚未得到服务确认，请重试软件急停。网络停止未确认时，请使用控制柜物理急停。'
        : occupied() || pausePending || estopPending || status?.inference_busy ? '等待当前操作和推理停止后可恢复。请先解除物理急停并检查现场。'
        : '先解除控制柜物理急停并检查现场，再勾选确认并手动恢复。恢复后保持暂停，不会自动使能或开始推理。');
      text('estopResetFeedback', estopResetFeedback);
      $('estopResetFeedback').hidden = !estopResetFeedback;
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
      const diagnostic = s.model_diagnostics || {};
      const remote = (s.inference_device || settings?.inference_device) === 'qizhi_h200';
      const health = diagnostic.remote || {}, remoteDiagnostics = health.model_diagnostics || {};
      text('deviceHeading', remote ? '远端推理设备' : '本机推理设备');
      if (remote) {
        const failed = s.state === 'error' || Boolean(diagnostic.error);
        const ready = online && s.model_ready && health.ready === true && !failed;
        badge('cudaBadge', !online ? '连接中断' : failed ? '远端需检查' : ready ? '远端已就绪' : s.state === 'loading' ? '正在连接' : '未连接', !online || failed ? 'bad' : ready ? 'good' : 'warn');
        text('cuda', health.gpu_name || health.model_name || (ready ? health.device : '') || '启智 H200 · 等待模型握手');
        text('cudaDetail', [health.device, health.dtype || remoteDiagnostics.dtype, health.inference_backend || remoteDiagnostics.inference_backend].filter(Boolean).join(' · ') || '连接远端后读取实际 GPU、精度与计算后端');
      } else {
        badge('cudaBadge', s.cuda?.available ? '可用' : s.cuda?.checking ? '检查中' : '不可用', s.cuda?.available ? 'good' : s.cuda?.checking ? 'warn' : 'bad');
        text('cuda', s.cuda?.available ? `${s.cuda.device}${s.cuda.total_bytes ? ` · ${(s.cuda.total_bytes / 1e9).toFixed(1)} GB` : ''}` : s.cuda?.checking ? '正在检查 CUDA 驱动' : '请检查 CUDA 环境与驱动');
        text('cudaDetail', [diagnostic.device, diagnostic.dtype, diagnostic.inference_backend].filter(Boolean).join(' · ') || '精度与计算后端在加载时确定');
      }
      $('remoteIdentity').hidden = !remote || !s.model_ready || health.ready !== true;
      text('remoteModelName', health.checkpoint_name || '远端 checkpoint');
      text('remoteModelHash', health.checkpoint_sha256 ? `SHA256 ${health.checkpoint_sha256}` : 'SHA256 待确认');
      const remoteProfile = health.sampling_profile === '4416' ? '4 / 4 / 16' : health.sampling_profile === '161616' ? '16 / 16 / 16' : health.sampling_profile;
      const optimization = typeof health.optimization === 'string' ? health.optimization
        : health.optimization?.effective_mode || health.optimization?.mode ||
          (health.optimization?.requested_mode ? `请求 ${health.optimization.requested_mode}` : null);
      const graphScope = health.optimization?.graph_scope;
      text('remoteModelConfig', [remoteProfile ? `采样 ${remoteProfile}` : null, health.action_space, optimization,
        graphScope ? `图捕获 ${graphScope}` : null].filter(Boolean).join(' · ') || '等待远端配置');
      badge('modelBadge', s.model_ready ? '模型已加载' : s.state === 'loading' ? '加载中' : '未加载', s.model_ready ? 'good' : 'warn');
      text('modelMode', `双视角观测 · ${s.model_ready ? (s.reference_mode === 'none' ? '无参考轨迹' : '滚动 AE 参考') : '模型未加载'} · 30 Hz 动作接口`);
      const profile = s.sampling_profile || settings?.sampling_profile;
      badge('activeProfile', settings?.supported === false ? '旧版采样配置' : profile === '4416' ? '4 / 4 / 16 · 平衡' : profile === '161616' ? '16 / 16 / 16 · 基准' : '采样设置待确认');
      text('executionMode', s.execution_mode === 'synchronous' ? '同步推理' : s.execution_mode === 'async_relaxed' ? '异步推理' : '异步严格时序');
      text('executionPhase', phases[s.execution_phase] || phases[s.state] || s.state);
      text('inferencePhase', s.inference_busy ? '模型推理中' : '模型空闲');
      const timing = diagnostic.last_inference || {};
      text('latencyLabel', remote ? '观测发送至 action 返回' : '模型延迟');
      text('latency', milliseconds(remote ? timing.elapsed_ms ?? s.latency_ms : s.latency_ms));
      $('remoteMetrics').hidden = !remote;
      text('serverLatency', milliseconds(timing.server_timing_ms?.inference));
      text('transportLatency', milliseconds(timing.transport_overhead_ms));
      text('encodeLatency', milliseconds(timing.encode_ms));
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
      text('diagnostics', JSON.stringify({inference_device: s.inference_device || 'local', model: diagnostic,
        local_cuda: s.cuda, hardware: s.hardware, gripper: gripperDebug}, null, 2));
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
          if (serverEstop()) {
            localEstop = true;
            if (estopResetFeedback.startsWith('急停已恢复')) {
              estopResetFeedback = '检测到新的软件急停，继续保持急停锁定。';
              $('estopResetConfirmed').checked = false;
            }
          }
          text('updated', `更新于 ${now().toLocaleTimeString('zh-CN', {hour12: false})}`);
          return value;
        } catch (error) {
          online = false; statusError = error.message;
          badge('badge', '服务连接中断', 'bad');
          return null;
        } finally { renderStatus(); }
      })();
      statusPromise = task;
      try { return await task; } finally { if (statusPromise === task) statusPromise = null; }
    }
    async function refreshCheckpoints() {
      if (catalogPromise) return catalogPromise;
      const task = (async () => {
        try {
          const value = await api('/api/checkpoints');
          if (!Array.isArray(value.checkpoints)) throw new Error('checkpoint 列表格式错误');
          checkpoints = value.checkpoints;
          catalogDirectory = value.directory || '未提供';
          const select = $('checkpoint'), previous = select.value;
          select.replaceChildren();
          for (const checkpoint of checkpoints) {
            const option = document.createElement('option'); option.value = checkpoint.id;
            option.textContent = `${checkpoint.label || checkpoint.id} · ${checkpoint.installed ? '本地文件' : '待下载'}`;
            select.append(option);
          }
          if (checkpoints.some(item => item.id === previous)) select.value = previous;
          else if (checkpoints.length) select.value = checkpoints[0].id;
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
          if (!preserveDraft) {
            draft = normalisedSettings(value);
            smoothingByDevice[draft.inference_device] = draft.action_smoothing_alpha;
            resultAgeByDevice[draft.inference_device] = draft.max_result_age_s;
            deviceTouched = false; remoteTouched = false; smoothingTouched = false;
          }
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
      if (emergency ? estopPending || (serverEstop() && !resettingEstop()) : pausePending || localEstop || resettingEstop()) return;
      if (emergency) {
        estopPending = true; localEstop = true; estopGeneration += 1;
        $('estopResetConfirmed').checked = false; estopResetError = '';
        estopResetFeedback = resettingEstop() ? '恢复已被新的软件急停打断，继续保持急停锁定。' : '';
        $('estopNotice').hidden = false;
      }
      else pausePending = true;
      actionError = ''; renderButtons(); showErrors();
      try { await api(emergency ? '/api/estop' : '/api/pause', {}, 10000); stopError = ''; }
      catch (error) { stopError = `${emergency ? '急停' : '暂停'}未确认：${error.message}${emergency ? ' 请使用控制柜物理急停。' : ''}`; }
      finally {
        if (emergency) estopPending = false; else pausePending = false;
        await refreshStatus(true); renderButtons(); showErrors();
      }
    }
    async function resetEstop() {
      if ($('estopReset').disabled || resettingEstop()) return;
      const generation = estopGeneration;
      estopResetPending = true; localEstop = true; estopResetError = '';
      estopResetFeedback = '正在恢复，请等待服务确认。';
      renderButtons(); showErrors();
      const resetConfirmed = value => value?.state === 'paused' && value.estop_latched === false &&
        value.estop_reset_busy === false && Number.isInteger(value.epoch);
      try {
        const result = await api('/api/estop/reset', {confirmed: true}, 30000);
        if (generation !== estopGeneration) return;
        if (!resetConfirmed(result)) throw new Error('服务尚未确认急停解除并保持暂停。');
        // Only a fresh status after the successful reset may release this browser's latch.
        const confirmed = await refreshStatus(true);
        if (generation !== estopGeneration) return;
        if (!online || !resetConfirmed(confirmed) || confirmed.epoch !== result.epoch) {
          throw new Error('恢复后的状态未确认或已发生变化，请刷新状态后重试。');
        }
        localEstop = false; stopError = '';
        $('estopResetConfirmed').checked = false;
        estopResetFeedback = '急停已恢复，保持暂停。确认设备就绪后再手动开始推理。';
      } catch (error) {
        if (generation === estopGeneration) {
          estopResetError = `恢复急停未确认：${error.message}`;
          estopResetFeedback = '恢复未确认，继续保持急停锁定。请检查状态后重试。';
        }
      } finally {
        estopResetPending = false;
        await refreshStatus(true); renderButtons(); showErrors();
      }
    }
    $('checkpoint').onchange = renderButtons;
    $('inferenceDevice').onchange = () => {
      const next = $('inferenceDevice').value;
      if (!['local', 'qizhi_h200'].includes(next)) return;
      smoothingByDevice[draft.inference_device] = draft.action_smoothing_alpha;
      resultAgeByDevice[draft.inference_device] = draft.max_result_age_s;
      draft.inference_device = next;
      draft.action_smoothing_alpha = smoothingByDevice[next];
      draft.max_result_age_s = resultAgeByDevice[next];
      deviceTouched = true; smoothingTouched = true;
      text('settingsFeedback', ''); renderButtons(); showErrors();
    };
    const updateRemoteDraft = () => {
      draft.remote_inference = {...remoteConfig(draft), url: $('remoteUrl').value.trim(),
        checkpoint_sha256: $('remoteCheckpointSha').value.trim().toLowerCase()};
      remoteTouched = true;
      text('settingsFeedback', ''); renderButtons();
    };
    $('remoteUrl').onchange = updateRemoteDraft;
    $('remoteCheckpointSha').onchange = updateRemoteDraft;
    $('remoteUrl').oninput = updateRemoteDraft;
    $('remoteCheckpointSha').oninput = updateRemoteDraft;
    $('smoothingAlpha').onchange = () => {
      draft.action_smoothing_alpha = $('smoothingAlpha').value;
      smoothingTouched = true;
      text('settingsFeedback', ''); renderButtons();
    };
    $('smoothingAlpha').oninput = $('smoothingAlpha').onchange;
    $('samplingProfile').onchange = () => {
      draft.sampling_profile = $('samplingProfile').value;
      if (draft.sampling_profile !== '4416') draft.execution_mode = 'synchronous';
      text('settingsFeedback', ''); renderButtons();
    };
    $('executionSelect').onchange = () => {
      draft.execution_mode = draft.sampling_profile === '4416' ? $('executionSelect').value : 'synchronous';
      text('settingsFeedback', ''); renderButtons();
    };
    $('applySettings').onclick = () => {
      if ($('applySettings').disabled || occupied()) return;
      const alpha = Number(draft.action_smoothing_alpha), remote = remoteConfig(draft);
      if (!String(draft.action_smoothing_alpha).trim() || !Number.isFinite(alpha) || alpha <= 0 || alpha > 1) {
        actionError = '动作平滑系数必须大于 0 且不超过 1；1 表示关闭平滑。'; showErrors(); return;
      }
      if (remoteMode()) {
        let address;
        try { address = new URL(remote.url); } catch (_) { /* Report one actionable validation error below. */ }
        if (!address || !['http:', 'https:'].includes(address.protocol) || !address.hostname || address.username || address.password) {
          actionError = '请填写有效的 HTTP / HTTPS 服务地址；认证令牌由服务启动环境提供。'; showErrors(); return;
        }
        if (remote.checkpoint_sha256 && !/^[a-f0-9]{64}$/.test(remote.checkpoint_sha256)) {
          actionError = '权重 SHA256 应为 64 位十六进制字符，或留空使用所选模型的校验值。'; showErrors(); return;
        }
      }
      return action('applySettings', '/api/settings', settingsPayload(), async () => {
        // Refresh the persisted settings before enabling the next model load.
        settings = null; await refreshSettings();
        text('settingsFeedback', settings ? `已应用，请${remoteMode() ? '重新连接远端模型' : '重新加载模型'}。` : '请刷新确认设置。');
      });
    };
    $('refresh').onclick = refreshCheckpoints;
    $('reload').onclick = () => Promise.all([refreshStatus(true), refreshSettings()]);
    $('download').onclick = () => action('download', '/api/download', {id: $('checkpoint').value}, async result => {
      text('downloadProgress', `下载完成：${result.path}`); await refreshCheckpoints();
    });
    $('load').onclick = () => action('load', '/api/select', {
      id: remoteMode() && remoteConfig(draft).checkpoint_sha256.trim() ? '' : $('checkpoint').value,
    });
    $('connect').onclick = () => action('connect', '/api/connect');
    $('start').onclick = () => action('start', '/api/start');
    $('pause').onclick = () => stop(false);
    $('estop').onclick = () => stop(true);
    $('estopResetConfirmed').onchange = renderButtons;
    $('estopReset').onclick = resetEstop;
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
