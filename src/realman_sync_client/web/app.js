"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const state = { payload: null, lastSuccess: 0, statusError: "", pending: 0, camera: {}, actionKey: "", eventKey: "", chartKey: "", samplingDirty: false, samplingOptionsKey: null, timingDirty: false, timingOptionsKey: null };
  const clientKey = document.querySelector('meta[name="client-key"]')?.content || "";
  const cameraNames = ["global", "wrist"];
  const gripperActions = {
    gripper_enable: {id:"gripper-enable-button", available:"enable_available", label:"使能"},
    gripper_disable: {id:"gripper-disable-button", available:"disable_available", label:"失能"},
    gripper_open: {id:"gripper-open-button", available:"motion_available", label:"张开"},
    gripper_close: {id:"gripper-close-button", available:"motion_available", label:"闭合"}
  };
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const hasResetPose = (pose) => !!pose.saved && Array.isArray(pose.joint_degrees) && pose.joint_degrees.length === 7 && pose.joint_degrees.every(finite) && finite(pose.speed_percent) && pose.speed_percent > 0 && pose.speed_percent <= 100 && typeof pose.recorded_at === "string" && !!pose.recorded_at;
  const number = (value, digits = 0) => finite(value) ? value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
  const text = (id, value) => { const element = byId(id); const output = String(value ?? "—"); if (element.textContent !== output) element.textContent = output; };
  const errorText = (id, value) => { const element = byId(id); element.hidden = !value; element.textContent = value || ""; };
  const fresh = () => state.lastSuccess > 0 && !state.statusError && performance.now() - state.lastSuccess < 1800;
  const labels = { disconnected: "设备未连接", offline: "离线", idle: "待机", starting: "正在启动", connecting: "正在连接", monitoring: "监测中", collecting: "采集中", observing: "采集中", inferring: "推理中", requesting: "请求中", executing: "动作执行中", settling: "等待关节停稳", stopping: "正在停止", stopped: "已停止", error: "需要处理", ready: "已就绪", paused: "已暂停", blocked: "等待执行条件", waiting_sensors: "等待传感器" };
  const phaseName = (value) => labels[value] || value || "待机";
  const pill = (id, label, tone = "muted") => { const element = byId(id); element.textContent = label; element.className = `pill ${tone}`; };

  function samplingProfiles(sampling) {
    const profiles = new Map();
    if (Array.isArray(sampling?.profiles)) sampling.profiles.forEach((profile) => {
      if (profile && typeof profile === "object" && !Array.isArray(profile) &&
          typeof profile.id === "string" && profile.id.trim() && !profiles.has(profile.id)) profiles.set(profile.id, profile);
    });
    return profiles;
  }
  const profileLabel = (profile) => typeof profile.label === "string" && profile.label.trim() ? profile.label : profile.id;

  function makeValues(id, names, units) {
    const fragment = document.createDocumentFragment();
    names.forEach((name, index) => {
      const cell = document.createElement("div"); cell.className = "value-cell";
      const label = document.createElement("div"); label.className = "value-label"; label.textContent = name;
      const unit = document.createElement("small"); unit.textContent = units[index] || ""; label.append(unit);
      const value = document.createElement("div"); value.className = "value-number"; value.textContent = "—"; value.dataset.index = String(index);
      cell.append(label, value); fragment.append(cell);
    });
    byId(id).replaceChildren(fragment);
  }
  makeValues("state-values", ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "夹爪"], ["rad", "rad", "rad", "rad", "rad", "rad", "rad", "0–1"]);
  makeValues("force-values", ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], ["N", "N", "N", "Nm", "Nm", "Nm"]);

  function updateValues(id, values) {
    const array = Array.isArray(values) ? values : [];
    byId(id).querySelectorAll(".value-number").forEach((element, index) => { element.textContent = number(array[index], 3); });
  }

  function renderButtons() {
    const data = state.payload || {}, inference = data.inference || {}, execution = data.execution || {}, sampling = data.sampling || {}, recovery = data.recovery || {};
    const gripper = data.gripper || {};
    const timing = data.timing || {};
    const resetPose = data.reset_pose || {};
    const active = fresh(), busy = state.pending;
    const button = byId("inference-button");
    button.dataset.action = inference.enabled ? "inference_stop" : "inference_start";
    button.textContent = inference.enabled ? "暂停自动推理" : "开始自动推理";
    button.disabled = busy || !active || !!sampling.busy || (!inference.enabled && !data.server?.ready);
    byId("execute-button").disabled = busy || !active || !!sampling.busy || !execution.available || !!execution.enabled;
    byId("stop-button").disabled = busy || !active || !(execution.enabled || recovery.resume_execution);
    byId("connect-button").disabled = busy || !active || data.phase === "connecting";
    byId("disconnect-button").disabled = busy || !active;
    const clearButton = byId("gripper-clear-button");
    clearButton.disabled = busy || !active || !gripper.clear_available || !!execution.enabled || !!gripper.clearing || !!gripper.busy;
    clearButton.textContent = gripper.clearing ? "正在清错并重连…" : "清除错误并重连";
    clearButton.title = !active ? "客户端状态未连接" : execution.enabled ? "请先停止机械臂执行" : gripper.clear_reason || "";
    const reconnectButton = byId("gripper-reconnect-button");
    reconnectButton.disabled = !!busy || !active || !!execution.enabled || !!sampling.busy || !!gripper.busy || !!gripper.reconnecting || !gripper.reconnect_available;
    reconnectButton.textContent = gripper.reconnecting ? "重连中…" : "重连通信";
    reconnectButton.title = !active ? "客户端状态未连接" : execution.enabled ? "请先停止动作执行" : gripper.reconnect_reason || "只读重连通信，不使能、不开合、不清错";
    Object.values(gripperActions).forEach((item) => {
      const controlButton = byId(item.id);
      controlButton.disabled = !!busy || !active || !!execution.enabled || !!gripper.clearing || !!gripper.busy || !gripper[item.available];
      controlButton.title = !active ? "客户端状态未连接" : execution.enabled ? "请先停止动作执行" : busy || gripper.clearing || gripper.busy ? "正在处理夹爪操作" : !gripper[item.available] ? "等待夹爪条件满足" : "";
    });
    const profiles = samplingProfiles(sampling);
    const samplingBlocked = !!busy || !active || !sampling.supported || !profiles.size || !!sampling.busy || !!execution.enabled || !!gripper.busy;
    const samplingSelect = byId("sampling-select"), applyButton = byId("sampling-apply-button");
    samplingSelect.disabled = samplingBlocked;
    applyButton.disabled = samplingBlocked || !profiles.has(samplingSelect.value) || (samplingSelect.value === sampling.sampling_profile && !sampling.error);
    applyButton.textContent = sampling.busy ? "切换中…" : "应用";
    applyButton.title = !active ? "客户端状态未连接" : execution.enabled ? "请先停止动作执行" : sampling.busy ? "正在切换并预热" : !sampling.supported || !profiles.size ? "当前 Server 未声明可切换档位" : samplingSelect.value === sampling.sampling_profile && !sampling.error ? "当前档位已生效" : "应用 Server 声明的采样档位";
    const timingProfiles = samplingProfiles(timing), timingSelect = byId("timing-select");
    const timingBlocked = !!busy || !active || !timing.supported || !timingProfiles.size ||
      !!timing.busy || !!sampling.busy || !!execution.enabled || !!gripper.busy || !!gripper.clearing ||
      !!gripper.reconnecting || !!gripper.auto_reconnect?.active || !!resetPose.busy || data.phase === "connecting";
    timingSelect.disabled = timingBlocked;
    byId("timing-apply-button").disabled = timingBlocked || !timingProfiles.has(timingSelect.value) ||
      (timingSelect.value === timing.profile && !timing.error);
    byId("timing-apply-button").textContent = timing.busy ? "切换中…" : "应用";
    const resetButton = byId("reset-pose-button");
    resetButton.disabled = !!busy || !active || !hasResetPose(resetPose) || !resetPose.available || !!resetPose.busy;
    resetButton.textContent = resetPose.busy ? "复位中…" : "复位到记录位置";
    resetButton.title = !active ? "客户端状态未连接" : resetPose.reason || (hasResetPose(resetPose) ? "先暂停当前推理和执行，再复位到记录位置；夹爪保持" : "缺少完整的记录位置");
    if (resetPose.busy || timing.busy) {
      document.querySelectorAll("button[data-action]").forEach((controlButton) => {
        if (!["execute_stop", "emergency_stop"].includes(controlButton.dataset.action)) controlButton.disabled = true;
      });
      samplingSelect.disabled = true;
      timingSelect.disabled = true;
      if (resetPose.busy) byId("stop-button").disabled = !active;
    }
    // Stop must remain usable even if another request is in flight or telemetry is stale.
    byId("emergency-button").disabled = false;
  }

  function renderConnection() {
    const connected = fresh();
    document.body.classList.toggle("offline", !connected);
    const element = byId("connection-label");
    element.querySelector(".status-dot").className = `status-dot ${connected ? "good" : "bad"}`;
    element.querySelector("span").textContent = connected ? "客户端在线" : state.lastSuccess ? "连接中断" : "正在连接";
    byId("footer-dot").className = `status-dot ${connected ? "good" : "bad"}`;
    const age = state.lastSuccess ? (performance.now() - state.lastSuccess) / 1000 : null;
    text("last-update", connected ? "实时状态已同步" : age !== null ? `状态已 ${number(age, 1)} 秒未更新 · 当前显示为旧数据` : "等待首次状态更新");
    const alert = byId("connection-alert");
    alert.hidden = !state.statusError && connected;
    alert.textContent = state.statusError ? `客户端状态连接异常：${state.statusError}${state.lastSuccess ? "。已保留最后数据，不能视为实时反馈。" : ""}` : connected ? "" : "正在连接本地客户端…";
    if (!connected) {
      cameraNames.forEach((name) => { document.querySelector(`[data-camera="${name}"]`).classList.add("stale"); errorText(`${name}-overlay`, "状态连接中断 · 保留最后画面"); });
      pill("gripper-status", state.lastSuccess ? "状态未更新" : "等待反馈", state.lastSuccess ? "bad" : "muted");
    }
    renderButtons();
  }

  function render(data) {
    const server = data.server || {}, sensors = data.sensors || {}, inference = data.inference || {}, execution = data.execution || {}, recovery = data.recovery || {};
    pill("mode-badge", data.mode === "mock" ? "模拟数据" : data.mode === "realman" ? "真机采集" : data.mode || "本地客户端", data.mode === "mock" ? "warning" : "muted");
    text("phase-label", recovery.active ? "等待反馈恢复" : phaseName(data.phase));
    for (const name of ["state", "force"]) {
      const sensor = sensors[name] || {};
      text(`${name}-hz`, number(sensor.hz, 1));
      text(`${name}-age`, finite(sensor.age_ms) ? `更新于 ${number(sensor.age_ms)} ms 前` : "等待数据");
      text(`${name}-samples`, `${number(sensor.samples)} 次`);
      updateValues(`${name}-values`, sensor.values);
      errorText(`${name}-error`, sensor.error);
    }
    const forceKind = sensors.force?.kind;
    const forceNames = { raw: "原始力反馈", filtered: "滤波力反馈", model: "模型输入力", mock: "模拟力反馈", calibrated: "校准力反馈", calibrated_filtered: "校准滤波力", controller_zero_force: "控制器原始力" };
    pill("force-kind", forceNames[forceKind] || forceKind || "等待反馈", ["raw", "controller_zero_force"].includes(forceKind) ? "warning" : "muted");
    const feedbackTiming = sensors.force?.callback_timing;
    text("force-timing", feedbackTiming ? `UDP 回调间隔 P95 ${number(feedbackTiming.arrival_gap_ms?.p95, 2)} ms · 处理 P95 ${number(feedbackTiming.processing_ms?.p95, 2)} ms · 超过 ${number(feedbackTiming.max_gap_ms)} ms 的空档 ${number(feedbackTiming.gaps_over_budget)} 次` : "等待 UDP 时序统计");
    text("inference-latency", number(inference.inference_ms));
    text("roundtrip-latency", number(inference.round_trip_ms));
    text("request-count", `${number(inference.requests_completed ?? 0)} 次完成`);
    text("source-age", finite(inference.obs_age_ms) ? `观测年龄 ${number(inference.obs_age_ms)} ms` : "观测年龄 —");
    pill("server-status", server.ready ? "在线" : server.error ? "未就绪" : "检测中", server.ready ? "good" : server.error ? "bad" : "muted");
    text("server-url", server.url); text("server-backend", server.backend); text("server-ping", finite(server.ping_ms) ? `${number(server.ping_ms)} ms` : "—");
    errorText("server-error", server.error);
    pill("inference-status", recovery.active ? "等待自动恢复" : inference.enabled ? phaseName(inference.phase || "inferring") : "已暂停", recovery.active ? "warning" : inference.enabled ? "good" : "muted");
    errorText("inference-error", inference.last_error);
    pill("execution-status", recovery.resume_execution ? "等待恢复执行" : execution.enabled ? "正在执行" : "未执行", recovery.resume_execution || execution.enabled ? "warning" : "muted");
    text("execution-blocked", recovery.resume_execution ? "等待反馈恢复后自动继续执行；点击“停止执行”可取消待恢复执行。" : execution.enabled ? "执行已开启，可随时停止。" : execution.blocked_reason || (execution.available ? "执行条件已满足，等待手动开始。" : "等待设备与执行条件检查"));
    const recoveryNotice = byId("recovery-notice");
    recoveryNotice.hidden = !recovery.active;
    recoveryNotice.textContent = recovery.active ? `等待反馈恢复，将自动继续。${recovery.resume_execution ? "反馈恢复后将恢复推理和动作执行；点击“停止执行”可取消待恢复执行。" : "反馈恢复后将恢复自动推理，不开启动作执行。"}${recovery.reason ? ` 原因：${recovery.reason}` : ""}${finite(recovery.attempts) && recovery.attempts > 0 ? ` 已尝试 ${number(recovery.attempts)} 次。` : ""}` : "";
    renderGripper(data.gripper || {}, execution);
    renderSampling(data.sampling || {});
    renderTiming(data.timing || {});
    renderResetPose(data.reset_pose || {});
    const cameras = Array.isArray(sensors.cameras) ? sensors.cameras : [];
    cameraNames.forEach((name) => {
      const camera = cameras.find((item) => item.name === name) || {};
      const preview = state.camera[name];
      const cameraStale = !camera.ready || !!preview.error;
      document.querySelector(`[data-camera="${name}"]`).classList.toggle("stale", cameraStale);
      pill(`${name}-status`, camera.ready ? preview.error ? "预览异常" : "采集中" : camera.error ? "反馈异常" : "等待画面", camera.ready && !preview.error ? "good" : camera.error || preview.error ? "warning" : "muted");
      text(`${name}-fps`, number(camera.fps, 1)); text(`${name}-frame`, number(camera.frame_number)); text(`${name}-age`, number(camera.age_ms));
      text(`${name}-serial`, `序列号 ${camera.serial || "—"}`);
      text(`${name}-resolution`, camera.width && camera.height ? `${camera.width} × ${camera.height}` : "640 × 480");
      errorText(`${name}-error`, camera.error || preview.error);
      errorText(`${name}-overlay`, cameraStale && preview.url ? "画面暂未更新 · 保留最后一帧" : null);
    });
    renderActions(inference.actions); renderChart(inference.latencies); renderEvents(data.events); renderConnection();
  }

  function renderResetPose(pose) {
    text("reset-pose-detail", finite(pose.speed_percent) ? `${number(pose.speed_percent)}% 速度 · 夹爪保持` : "等待记录位置 · 夹爪保持");
    const recorded = pose.recorded_at ? new Date(pose.recorded_at) : null;
    text("reset-pose-recorded", `记录时间：${recorded && !Number.isNaN(recorded.getTime()) ? recorded.toLocaleString("zh-CN", {hour12:false}) : pose.recorded_at || "—"}`);
    text("reset-pose-reason", pose.busy ? "复位中，可随时点击停止执行。" : pose.reason || (hasResetPose(pose) ? "点击后先暂停当前推理和执行，再以记录速度复位。" : "尚未取得完整的复位配置"));
    const result = pose.last_result || {};
    errorText("reset-pose-result", result.message || result.error);
    if (pose.busy) {
      pill("execution-status", "复位中", "warning");
      text("phase-label", "复位中");
      text("execution-blocked", "正在返回记录位置；停止执行可中止复位。" );
    }
  }

  function renderGripper(gripper, execution) {
    const code = gripper.fault_code;
    const known = Number.isInteger(code) && code >= 0;
    const disabledHint = known && code === 0 && gripper.enabled === false && /^(夹爪)?未使能(?:，请点击夹爪使能)?[。.!！\s]*$/.test(String(gripper.error || ""));
    const controlLimit = finite(gripper.control_max_age_ms) ? gripper.control_max_age_ms : 500;
    const observationLimit = finite(gripper.observation_max_age_ms) ? gripper.observation_max_age_ms : 1000;
    const feedbackStale = (finite(gripper.age_ms) && gripper.age_ms > controlLimit) ||
      (finite(gripper.enabled_age_ms) && gripper.enabled_age_ms > controlLimit) || /过期/.test(String(gripper.error || ""));
    const transport = gripper.transport || {};
    const reconnect = gripper.auto_reconnect || {};
    const transportName = ({json:"TCP JSON / 末端 Modbus", sdk:"SDK / 末端 Modbus"})[transport.backend] || transport.backend || "等待通路";
    const linkState = reconnect.active ? "只读自动重连中" :
      reconnect.phase === "failed" ? "自动重连已停止，请检查连接" :
      transport.failed || transport.closed ? "连接已关闭，需重连" :
      transport.needs_sync ? "超时后同步恢复中" : gripper.communication_error ? "读取异常" :
      finite(gripper.age_ms) && gripper.age_ms <= observationLimit && finite(gripper.samples) && gripper.samples > 0 ? "反馈正在更新" : "等待有效反馈";
    text("gripper-link-state", `通信状态：${linkState}${gripper.freshness_error ? ` · ${gripper.freshness_error}` : ""}${known && code !== 0 ? " · 设备报警仍需单独处理" : ""}`);
    text("gripper-opening", finite(gripper.opening) ? `${number(gripper.opening * 100, 1)}%` : "—");
    text("gripper-target", finite(gripper.requested_opening) ? `${number(gripper.requested_opening * 100, 1)}%` : "—");
    text("gripper-enabled", gripper.enabled === true ? "已使能" : gripper.enabled === false ? "已失能" : "未知");
    text("gripper-age", finite(gripper.age_ms) ? `${number(gripper.age_ms)} ms` : "—");
    text("gripper-hz", finite(gripper.hz) ? `${number(gripper.hz, 1)} Hz` : "—");
    text("gripper-interval", finite(gripper.last_interval_ms) ? `${number(gripper.last_interval_ms)} ms` : "—");
    text("gripper-cadence", `近 2 秒最长间隔 ${number(gripper.max_interval_ms)} ms · ${number(gripper.samples)} 次有效反馈 · 推理观测 ≤ ${number(observationLimit)} ms · 动作反馈 ≤ ${number(controlLimit)} ms`);
    text("gripper-communication", `${transportName} · 位置读取 ${number(gripper.read_duration_ms, 1)} ms · 使能年龄 ${number(gripper.enabled_age_ms)} ms · 使能轮询 ${number(finite(gripper.enable_poll_s) ? gripper.enable_poll_s * 1000 : null)} ms`);
    text("gripper-transport-stats", `成功事务 RTT P50 / P95 ${number(transport.p50_ms, 1)} / ${number(transport.p95_ms, 1)} ms · 全部尝试最长 ${number(transport.all_attempt_max_ms, 1)} ms · TCP 等待超时 ${number(transport.timeouts)} 次 · 外设读超时 ${number(transport.peripheral_read_timeouts || 0)} 次 · 只读重连 ${number(reconnect.count || 0)} 次 · 当前连接失败 ${number(transport.errors)} / ${number(transport.transactions)} 次`);
    text("gripper-fault", known ? `0x${code.toString(16).toUpperCase().padStart(4, "0")}${code === 0x20 ? " · 夹取掉落警报" : ""}` : "未知");
    text("gripper-reconnect-reason", gripper.reconnect_reason || "只读重连通信，不使能、不开合、不清错；完成后需手动恢复推理和执行。");
    if (gripper.reconnecting) pill("gripper-status", "通信重连中", "warning");
    else if (gripper.clearing) pill("gripper-status", "清除中", "warning");
    else if (gripper.busy) pill("gripper-status", `${gripperActions[gripper.operation]?.label || ({enable:"使能",disable:"失能",open:"张开",close:"闭合"}[gripper.operation]) || "操作"}中`, "warning");
    else if (known && code !== 0) pill("gripper-status", "有报警", "bad");
    else if (feedbackStale && gripper.measurement_ready) pill("gripper-status", "观测可用 · 动作等待反馈", "warning");
    else if (feedbackStale) pill("gripper-status", "观测反馈过期", "bad");
    else if (gripper.error && !disabledHint) pill("gripper-status", "反馈异常", "bad");
    else if (known && gripper.enabled === false) pill("gripper-status", "无报警 · 已失能", "muted");
    else if (known && gripper.enabled === true && gripper.ready) pill("gripper-status", "无报警", "good");
    else if (known && gripper.enabled === true) pill("gripper-status", "等待就绪", "warning");
    else if ((gripper.ready || gripper.measurement_ready) && known) pill("gripper-status", "无报警", "good");
    else pill("gripper-status", known ? "等待反馈" : "报警码未知", "muted");
    const reason = gripper.clearing ? "正在等待夹爪读回确认…" : execution.enabled ? "请先停止机械臂执行。" :
      gripper.clear_reason || (gripper.clear_available ? "先取下夹持物；重建通信并失能清错，完成后保持失能。" : "等待夹爪诊断");
    const recoveryHint = gripper.clear_available && !gripper.clearing && !execution.enabled ? " 同时恢复并读回采集端参数：速度 100%、力矩 50%；完成后仍失能，不发送开合目标。" : "";
    text("gripper-clear-reason", reason + recoveryHint);
    errorText("gripper-error", gripper.error);
    byId("gripper-error").classList.toggle("hint", disabledHint);
    const result = gripper.last_result || {};
    const resultElement = byId("gripper-result");
    resultElement.hidden = !result.message;
    resultElement.textContent = result.message ? `最近夹爪操作：${result.message}` : "";
    resultElement.classList.toggle("failed", result.ok === false || result.success === false);
  }

  function renderSampling(sampling) {
    const profile = sampling.sampling_profile;
    const profiles = samplingProfiles(sampling), supportedProfile = profiles.has(profile);
    const singleForward = sampling.video_mode === "single_forward";
    const actionOnly = sampling.model === "pi05" && sampling.force_conditioning === false;
    text("sampling-current", supportedProfile ? profileLabel(profiles.get(profile)) : profile || "—");
    text("sampling-steps", actionOnly ? `动作 ${number(sampling.action_steps)} 步；不使用力条件`
      : singleForward
      ? `视觉单次前向；Force/Action ${number(sampling.force_steps)}/${number(sampling.action_steps)}`
      : `视频 ${number(sampling.video_steps)} / 力 ${number(sampling.force_steps)} / 动作 ${number(sampling.action_steps)} 步`);
    text("sampling-label", actionOnly ? "π0.5 · 动作去噪步数" : singleForward ? "视觉单次前向 · Force / Action 去噪步数" : "去噪步数 · 视频 / 力 / 动作");
    if (sampling.busy) pill("sampling-status", "切换预热中", "warning");
    else if (sampling.error) pill("sampling-status", "切换失败", "bad");
    else if (!sampling.supported) pill("sampling-status", "暂不可用", "muted");
    else pill("sampling-status", supportedProfile ? "已生效" : "等待状态", supportedProfile ? "good" : "muted");
    const select = byId("sampling-select");
    const previous = select.value;
    if (!profiles.has(previous)) state.samplingDirty = false;
    const optionsKey = JSON.stringify([...profiles.values()].map((item) => [item.id, profileLabel(item)]));
    if (optionsKey !== state.samplingOptionsKey) {
      const fragment = document.createDocumentFragment();
      const placeholder = document.createElement("option");
      placeholder.value = ""; placeholder.textContent = profiles.size ? "选择 Server 采样档位" : "Server 未声明可切换档位";
      placeholder.disabled = true; fragment.append(placeholder);
      profiles.forEach((item) => {
        const option = document.createElement("option"); option.value = item.id; option.textContent = profileLabel(item); fragment.append(option);
      });
      select.replaceChildren(fragment);
      state.samplingOptionsKey = optionsKey;
    }
    select.value = state.samplingDirty && profiles.has(previous) ? previous : supportedProfile ? profile : "";
    if (supportedProfile && select.value === profile && !sampling.busy) state.samplingDirty = false;
    errorText("sampling-error", sampling.error);
  }

  function renderTiming(timing) {
    byId("timing-panel").hidden = !timing.supported;
    const profiles = samplingProfiles(timing), current = profiles.get(timing.profile);
    text("timing-current", current ? profileLabel(current) : timing.profile || "—");
    if (timing.busy) pill("timing-status", "切换连接中", "warning");
    else if (timing.error) pill("timing-status", "切换失败", "bad");
    else pill("timing-status", current ? "已生效" : "等待配置", current ? "good" : "muted");
    const limits = timing.limits || {};
    text("timing-limits", `夹爪动作 ${number(limits.gripper_control_max_age_s * 1000)} ms · 关节 ${number(limits.feedback_max_age_s * 1000)} ms · 相机年龄 ${number(limits.camera_max_age_s * 1000)} ms · 对齐 ${number(limits.camera_pair_skew_s * 1000)} ms`);
    text("timing-description", current?.description || "等待客户端时序配置");
    const select = byId("timing-select"), previous = select.value;
    if (!profiles.has(previous)) state.timingDirty = false;
    const key = JSON.stringify([...profiles.values()].map((item) => [item.id, profileLabel(item)]));
    if (key !== state.timingOptionsKey) {
      const fragment = document.createDocumentFragment();
      const placeholder = document.createElement("option");
      placeholder.value = ""; placeholder.textContent = "选择时序实验档位"; placeholder.disabled = true;
      fragment.append(placeholder);
      profiles.forEach((item) => {
        const option = document.createElement("option"); option.value = item.id;
        option.textContent = profileLabel(item); fragment.append(option);
      });
      select.replaceChildren(fragment);
      state.timingOptionsKey = key;
    }
    select.value = state.timingDirty && profiles.has(previous) ? previous : current ? timing.profile : "";
    if (current && select.value === timing.profile && !timing.busy) state.timingDirty = false;
    errorText("timing-error", timing.error);
  }

  function renderActions(actions) {
    const rows = Array.isArray(actions) ? actions : [];
    const key = JSON.stringify(rows);
    if (key === state.actionKey) return;
    state.actionKey = key;
    pill("action-shape", rows.length ? `${rows.length} × ${Array.isArray(rows[0]) ? rows[0].length : 0}` : "等待结果", rows.length ? "good" : "muted");
    const fragment = document.createDocumentFragment();
    if (!rows.length) {
      const row = document.createElement("tr"); row.className = "empty-row";
      const cell = document.createElement("td"); cell.colSpan = 9; cell.textContent = "尚无 action，完成一次推理后显示。"; row.append(cell); fragment.append(row);
    }
    rows.slice(0, 32).forEach((values, index) => {
      const row = document.createElement("tr"), step = document.createElement("td"); step.textContent = String(index + 1).padStart(2, "0"); row.append(step);
      for (let column = 0; column < 8; column += 1) { const cell = document.createElement("td"); cell.textContent = number(values?.[column], 3); row.append(cell); }
      fragment.append(row);
    });
    byId("action-rows").replaceChildren(fragment);
  }

  function svgNode(tag, attributes, content) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    if (content !== undefined) element.textContent = content;
    return element;
  }
  function renderChart(latencies) {
    const points = (Array.isArray(latencies) ? latencies : []).filter((item) => finite(item.inference_ms) && finite(item.round_trip_ms)).slice(-36);
    const key = JSON.stringify(points); if (key === state.chartKey) return; state.chartKey = key;
    const svg = byId("latency-chart"), fragment = document.createDocumentFragment();
    const left = 45, right = 410, top = 12, bottom = 117;
    const maximum = Math.max(100, ...points.flatMap((item) => [item.inference_ms, item.round_trip_ms])) * 1.15;
    [0, .5, 1].forEach((fraction) => {
      const y = bottom - fraction * (bottom - top);
      fragment.append(svgNode("line", {x1:left,y1:y,x2:right,y2:y,stroke:"#2a3543","stroke-dasharray":"3 5"}));
      fragment.append(svgNode("text", {x:left-8,y:y+3,"text-anchor":"end",fill:"#71859d","font-size":9,"font-family":"Consolas,monospace"}, number(maximum * fraction)));
    });
    fragment.append(svgNode("text", {x:7,y:10,fill:"#71859d","font-size":8}, "ms"));
    fragment.append(svgNode("text", {x:left,y:137,fill:"#71859d","font-size":8}, points.length ? `最近 ${points.length} 次` : "等待推理"));
    fragment.append(svgNode("text", {x:right,y:137,fill:"#71859d","font-size":8,"text-anchor":"end"}, "最新"));
    for (const [field, color] of [["round_trip_ms", "#8baefa"], ["inference_ms", "#66d1bb"]]) {
      const coordinates = points.map((point, index) => [points.length === 1 ? right : left + index / (points.length - 1) * (right - left), bottom - point[field] / maximum * (bottom - top)]);
      if (coordinates.length > 1) fragment.append(svgNode("polyline", {points:coordinates.map((p) => p.join(",")).join(" "),stroke:color,"stroke-width":1.7,fill:"none","stroke-linecap":"round","stroke-linejoin":"round"}));
      if (coordinates.length) { const last = coordinates.at(-1); fragment.append(svgNode("circle", {cx:last[0],cy:last[1],r:2.6,fill:color})); }
    }
    svg.replaceChildren(fragment); byId("chart-empty").hidden = points.length > 0;
  }

  function eventTime(value) {
    if (!value) return "—";
    const parsed = typeof value === "number" ? new Date(value < 1e12 ? value * 1000 : value) : new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value).slice(0, 12) : parsed.toLocaleTimeString("zh-CN", {hour12:false});
  }
  function renderEvents(events) {
    const entries = (Array.isArray(events) ? events : []).slice(-50).reverse();
    const key = JSON.stringify(entries); if (key === state.eventKey) return; state.eventKey = key;
    const fragment = document.createDocumentFragment();
    entries.forEach((event) => {
      const item = document.createElement("li"), level = String(event.level || "info").toLowerCase();
      item.className = level.includes("error") || level === "critical" ? "event-error" : level.includes("warn") ? "event-warning" : "event-info";
      const time = document.createElement("time"); time.textContent = eventTime(event.time);
      const dot = document.createElement("span"); dot.className = "event-dot";
      const message = document.createElement("span"); message.className = "event-message"; message.textContent = event.message || "";
      item.append(time, dot, message); fragment.append(item);
    });
    if (!entries.length) { const item = document.createElement("li"); item.className = "event-empty"; item.textContent = "等待客户端事件"; fragment.append(item); }
    byId("event-list").replaceChildren(fragment); text("event-count", `${entries.length} 条`);
  }

  async function requestJson(url, options = {}, timeoutMs = 5000) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {cache:"no-store", credentials:"same-origin", ...options, signal:controller.signal});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || result.message || `HTTP ${response.status}`);
      return result;
    } catch (error) { if (error.name === "AbortError") throw new Error("请求超时"); throw error; }
    finally { clearTimeout(timer); }
  }

  async function pollStatus() {
    const started = performance.now();
    try {
      const result = await requestJson("/api/status");
      if (!result || typeof result !== "object" || Array.isArray(result)) throw new Error("客户端返回的状态格式不正确");
      state.payload = result; state.lastSuccess = performance.now(); state.statusError = ""; render(result);
    } catch (error) { state.statusError = error.message || "状态读取失败"; renderConnection(); }
    finally { setTimeout(pollStatus, Math.max(0, (document.hidden ? 1000 : 250) - (performance.now() - started))); }
  }

  async function pollCamera(name) {
    const started = performance.now(), preview = state.camera[name];
    let nextDelay = 200;
    try {
      const camera = state.payload?.sensors?.cameras?.find((item) => item.name === name);
      if (document.hidden || !fresh() || !camera?.ready) { nextDelay = 700; return; }
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 2500);
      let blob;
      try {
        const response = await fetch(`/api/camera/${name}.jpg?t=${Date.now()}`, {cache:"no-store", credentials:"same-origin", signal:controller.signal});
        if (!response.ok) throw new Error(`预览不可用 (${response.status})`);
        blob = await response.blob();
        if (!blob.type.startsWith("image/") || !blob.size) throw new Error("尚无有效相机图像");
      } finally { clearTimeout(timer); }
      const newUrl = URL.createObjectURL(blob), image = byId(`${name}-image`), oldUrl = preview.url;
      try {
        const candidate = new Image(); candidate.src = newUrl;
        await candidate.decode();
      } catch (error) {
        URL.revokeObjectURL(newUrl);
        throw new Error("相机图像解码失败，已保留最后画面");
      }
      image.src = newUrl; image.hidden = false; byId(`${name}-placeholder`).hidden = true;
      preview.url = newUrl; preview.error = "";
      if (oldUrl) URL.revokeObjectURL(oldUrl);
    } catch (error) { preview.error = error.name === "AbortError" ? "相机预览请求超时" : error.message || "相机预览连接异常"; nextDelay = 1000; }
    finally { setTimeout(() => pollCamera(name), Math.max(0, nextDelay - (performance.now() - started))); }
  }

  async function control(button) {
    const action = button.dataset.action;
    const resetPose = state.payload?.reset_pose || {};
    const stopDuringReset = action === "execute_stop" && resetPose.busy && fresh();
    if (!action || (state.pending && action !== "emergency_stop" && !stopDuringReset)) return;
    if (resetPose.busy && !["execute_stop", "emergency_stop"].includes(action)) return;
    if (state.payload?.timing?.busy && !["execute_stop", "emergency_stop"].includes(action)) return;
    if (action === "reset_pose" && (!fresh() || !hasResetPose(resetPose) || !resetPose.available || resetPose.busy)) return;
    const payload = {action};
    if (action === "sampling_set") {
      const sampling = state.payload?.sampling || {};
      const profile = byId("sampling-select").value;
      if (!fresh() || !sampling.supported || sampling.busy || state.payload?.execution?.enabled || state.payload?.gripper?.busy || !samplingProfiles(sampling).has(profile)) return;
      payload.sampling_profile = profile;
    }
    if (action === "timing_set") {
      const timing = state.payload?.timing || {}, gripper = state.payload?.gripper || {};
      const profile = byId("timing-select").value;
      if (!fresh() || !timing.supported || timing.busy || state.payload?.sampling?.busy ||
          state.payload?.execution?.enabled || gripper.busy || gripper.clearing || gripper.reconnecting ||
          gripper.auto_reconnect?.active || state.payload?.phase === "connecting" ||
          !samplingProfiles(timing).has(profile)) return;
      payload.timing_profile = profile;
    }
    const gripperAction = gripperActions[action];
    if (gripperAction) {
      const gripper = state.payload?.gripper || {};
      if (!fresh() || !gripper[gripperAction.available] || gripper.clearing || gripper.busy || state.payload?.execution?.enabled) return;
    }
    if (action === "execute_start" && (!fresh() || !state.payload?.execution?.available)) return;
    if (action === "gripper_reconnect" && (!fresh() || !state.payload?.gripper?.reconnect_available || state.payload?.gripper?.reconnecting || state.payload?.gripper?.busy || state.payload?.sampling?.busy || state.payload?.execution?.enabled)) return;
    if (action === "gripper_clear_error" && (!fresh() || !state.payload?.gripper?.clear_available || state.payload?.gripper?.clearing || state.payload?.gripper?.busy || state.payload?.execution?.enabled)) return;
    state.pending += 1; renderButtons();
    const notice = byId("command-message"); notice.hidden = false; notice.className = "notice";
    notice.textContent = action === "emergency_stop" ? "正在发送紧急停止…" : action === "reset_pose" ? "正在提交复位请求，先暂停当前推理和执行；夹爪保持…" : action === "gripper_reconnect" ? "正在只读重连通信；不使能、不开合、不清错…" : action === "gripper_clear_error" ? "正在重建通信并失能清错，等待读回确认…" : action === "sampling_set" ? "正在提交采样档位，Client 将暂停推理并预热…" : "正在处理操作…";
    if (action === "timing_set") notice.textContent = "正在切换时序档位，Client 将暂停推理并重新连接；完成后手动开始推理和执行。";
    try {
      const result = await requestJson("/api/control", {method:"POST", headers:{"Content-Type":"application/json", "X-Client-Key":clientKey}, body:JSON.stringify(payload)}, 15000);
      notice.textContent = result.message || (gripperAction ? `夹爪${gripperAction.label}请求已提交，请查看设备反馈。` : {connect:"连接请求已提交",disconnect:"断开请求已提交",inference_start:"自动推理已开启",inference_stop:"自动推理已暂停",execute_start:"执行请求已提交",execute_stop:"停止执行请求已提交",emergency_stop:"紧急停止请求已提交",reset_pose:"复位请求已提交，请查看设备反馈。",gripper_clear_error:"清错与重连请求已提交，请查看夹爪反馈；尚未收到读回确认结果。",sampling_set:"采样切换已提交。等待预热完成后，请手动恢复自动推理。"}[action] || "操作已提交");
      if (action === "emergency_stop") notice.className = "notice warning";
    } catch (error) { notice.className = "notice error"; notice.textContent = `操作未完成：${error.message || "连接失败"}`; }
    finally { state.pending -= 1; renderButtons(); }
  }

  document.querySelectorAll("button[data-action]").forEach((button) => button.addEventListener("click", () => control(button)));
  byId("sampling-select").addEventListener("change", () => { state.samplingDirty = true; renderButtons(); });
  byId("timing-select").addEventListener("change", () => { state.timingDirty = true; renderButtons(); });
  cameraNames.forEach((name) => { state.camera[name] = {url:null,error:""}; pollCamera(name); });
  setInterval(() => { text("local-clock", new Date().toLocaleTimeString("zh-CN", {hour12:false})); renderConnection(); }, 500);
  window.addEventListener("pagehide", () => cameraNames.forEach((name) => { if (state.camera[name].url) URL.revokeObjectURL(state.camera[name].url); }));
  pollStatus();
})();
