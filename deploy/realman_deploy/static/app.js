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
    let draft = {sampling_profile: '161616', execution_mode: 'synchronous'};
    const text = (id, value) => { $(id).textContent = value; };
    function badge(id, label, kind = '') { text(id, label); $(id).className = `badge ${kind}`; }
    const inactive = () => online && status && !status.inference_busy && !status.gripper_busy &&
      ['idle', 'ready', 'paused', 'error'].includes(status.state) && !localEstop;
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
      const locked = !settings || settings.supported === false || busy || !inactive() || pausePending || estopPending;
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
      const locked = busy || !inactive() || pausePending || estopPending;
      $('checkpoint').disabled = busy || !checkpoints.length || !inactive();
      $('download').disabled = busy || !selected || selected.installed || selected.source === 'local' || localEstop;
      $('load').disabled = locked || !selected?.installed || !status?.cuda?.available || !settings || changed();
      $('connect').disabled = locked;
      const gripper = status?.hardware?.gripper;
      const gripperLocked = locked || !status?.hardware?.connected || !gripper?.initialized;
      $('gripperEnable').disabled = gripperLocked || gripper?.enabled === true;
      $('gripperDisable').disabled = gripperLocked || gripper?.enabled !== true;
      $('gripperOpen').disabled = gripperLocked || !gripper?.ready;
      $('gripperClose').disabled = gripperLocked || !gripper?.ready;
      $('gripperHold').disabled = gripperLocked;
      $('start').disabled = locked || !settings || !status?.model_ready || !status?.robot_ready || !status?.cameras_ready ||
        !status?.force_ready || !status?.gripper_ready || !status?.cuda?.available || changed() ||
        !['ready', 'paused', 'idle'].includes(status?.state);
      // Stop controls have their own lane: loading/downloading must not block them.
      $('pause').disabled = pausePending || estopPending || localEstop || status?.state === 'estop';
      $('pause').textContent = pausePending ? '正在暂停…' : status?.state === 'error' ? '复位错误 / 保持停止' : '暂停 / 缓停';
      $('estop').disabled = estopPending || status?.state === 'estop';
      $('estop').textContent = estopPending ? '正在急停…' : '软件急停';
      $('refresh').disabled = Boolean(catalogPromise) || busy;
      renderSettings();
    }
    function renderStatus() {
      const s = status;
      if (!s) { renderButtons(); showErrors(); return; }
      badge('badge', online ? (phases[s.state] || s.state) : '服务连接中断', !online || ['error', 'estop'].includes(s.state) ? 'bad' : s.state === 'running' ? 'good' : 'warn');
      badge('robot', s.robot_ready ? '就绪' : '未就绪', s.robot_ready ? 'good' : 'warn');
      badge('cameras', s.cameras_ready ? '双路就绪' : '未就绪', s.cameras_ready ? 'good' : 'warn');
      text('force', s.force_ready ? '就绪' : '未就绪');
      text('gripper', s.gripper_ready ? '就绪' : '未就绪');
      const gripper = s.hardware?.gripper;
      text('gripperDetail', gripper?.initialized
        ? `${gripper.enabled ? '已使能' : '未使能'} · 实际开度 ${Number.isFinite(gripper.opening) ? (gripper.opening * 100).toFixed(1) + '%' : '未知'} · 待发目标 ${Number.isFinite(gripper.requested_opening) ? (gripper.requested_opening * 100).toFixed(1) + '%' : '无'} · 已下发 ${Number.isFinite(gripper.last_sent?.opening) ? (gripper.last_sent.opening * 100).toFixed(1) + '%' : '未知'}${gripper.error ? ' · ' + gripper.error : ''}`
        : '先连接硬件，读取真实使能与开度');
      badge('cudaBadge', s.cuda?.available ? '可用' : s.cuda?.checking ? '检查中' : '不可用', s.cuda?.available ? 'good' : s.cuda?.checking ? 'warn' : 'bad');
      text('cuda', s.cuda?.available ? `${s.cuda.device}${s.cuda.total_bytes ? ` · ${(s.cuda.total_bytes / 1e9).toFixed(1)} GB` : ''}` : s.cuda?.checking ? '正在检查 CUDA 驱动' : '请检查 CUDA 环境与驱动');
      const diagnostic = s.model_diagnostics || {};
      const deviceInfo = diagnostic.device;
      const optimization = diagnostic.optimization || {};
      const optimizationLabel = optimization.effective_mode === 'compiled_cuda_graph'
        ? '编译＋完整采样图已启用' : optimization.effective_mode === 'cuda_graph'
        ? (optimization.graph_scope === 'complete_sampling_loop' ? '完整采样图已启用' : '分支采样图已启用')
        : optimization.effective_mode === 'pending' ? '推理优化准备中'
        : optimization.effective_mode ? `推理优化：${optimization.effective_mode}` : '';
      text('cudaDetail', [typeof deviceInfo === 'object' ? deviceInfo?.device : deviceInfo,
        deviceInfo?.dtype || diagnostic.dtype, diagnostic.inference_backend, optimizationLabel,
        optimization.fallback_reason, optimization.error].filter(Boolean).join(' · ') || '精度与计算后端在加载时确定');
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
      text('diagnostics', JSON.stringify({model: diagnostic, cuda: s.cuda, hardware: s.hardware}, null, 2));
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
      if (busy || $(id).disabled) return;
      busy = true; actionError = ''; const label = $(id).textContent;
      text(id, {connect: '正在连接…', load: '正在加载…', download: '正在下载…', applySettings: '正在应用…'}[id] || '处理中…');
      renderButtons(); showErrors();
      try {
        const result = await api(path, body);
        if (after) await after(result);
      } catch (error) { actionError = error.message; }
      finally { busy = false; text(id, label); await refreshStatus(true); renderButtons(); showErrors(); }
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
    for (const [id, operation] of Object.entries({gripperEnable: 'enable', gripperDisable: 'disable',
      gripperOpen: 'open', gripperClose: 'close', gripperHold: 'hold'})) {
      $(id).onclick = () => action(id, '/api/gripper', {operation});
    }
    $('start').onclick = () => action('start', '/api/start');
    $('pause').onclick = () => stop(false);
    $('estop').onclick = () => stop(true);
    async function init() {
      renderButtons();
      await Promise.all([refreshCheckpoints(), refreshSettings(), refreshStatus()]);
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
