"use strict";

// Offline execution of the complete production IIFE. No browser or live fetch.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const web = path.join(__dirname, "../src/realman_sync_client/web");
const source = fs.readFileSync(path.join(web, "app.js"), "utf8");
const html = fs.readFileSync(path.join(web, "index.html"), "utf8");

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.listeners = new Map();
    this.className = "";
    this.hidden = false;
    this.disabled = false;
    this._text = "";
    this._value = null;
    this.classList = {
      contains: (name) => this.className.split(/\s+/).includes(name),
      add: (...names) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), ...names])].join(" "); },
      remove: (...names) => { this.className = this.className.split(/\s+/).filter((name) => !names.includes(name)).join(" "); },
      toggle: (name, force) => {
        const add = force === undefined ? !this.classList.contains(name) : !!force;
        if (add) this.classList.add(name); else this.classList.remove(name);
        return add;
      }
    };
  }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get options() { return this.children.filter((child) => child.tagName === "OPTION"); }
  get value() {
    if (this.tagName === "SELECT") {
      if (this._value === null) return this.options[0]?.value || "";
      return this.options.some((option) => option.value === this._value) ? this._value : "";
    }
    return this._value === null ? this.textContent : this._value;
  }
  set value(value) {
    const string = String(value);
    this._value = this.tagName === "SELECT" && !this.options.some((option) => option.value === string) ? "" : string;
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node.tagName === "#FRAGMENT") { this.children.push(...node.children); node.children = []; }
      else this.children.push(node);
    }
  }
  replaceChildren(...nodes) {
    this.children = []; this._text = "";
    if (this.tagName === "SELECT") this._value = null;
    this.append(...nodes);
  }
  setAttribute(key, value) {
    if (key === "class") this.className = value;
    else if (key.startsWith("data-")) this.dataset[key.slice(5)] = value;
    else this[key] = value;
  }
  querySelectorAll(selector) {
    const found = [];
    for (const child of this.children) {
      if (selector.startsWith(".") ? child.classList.contains(selector.slice(1)) : child.tagName.toLowerCase() === selector) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }
  async dispatch(type) {
    // Like dispatchEvent: tests can attempt a synthetic click on a disabled control.
    await Promise.all((this.listeners.get(type) || []).map((listener) => listener({type, target:this})));
  }
}

function makeDocument() {
  const elements = new Map(), all = [];
  for (const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
    const attributes = Object.fromEntries([...match[2].matchAll(/([\w-]+)="([^"]*)"/g)].map((item) => [item[1], item[2]]));
    const element = new Element(match[1]);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    element.disabled = /(?:^|\s)disabled(?:\s|$)/.test(match[2]);
    all.push(element);
    if (attributes.id) elements.set(attributes.id, element);
  }
  const connection = elements.get("connection-label");
  const dot = new Element("i"); dot.className = "status-dot";
  connection.append(dot, new Element("span"));
  return {
    body:new Element("body"), hidden:false, elements,
    getElementById(id) { assert.ok(elements.has(id), `app referenced missing HTML id ${id}`); return elements.get(id); },
    createElement: (tag) => new Element(tag),
    createElementNS: (_namespace, tag) => new Element(tag),
    createDocumentFragment: () => new Element("#fragment"),
    querySelector(selector) {
      if (selector === 'meta[name="client-key"]') return {content:"offline-test-client-key"};
      const camera = selector.match(/^\[data-camera="([^"]+)"\]$/);
      if (camera) return all.find((element) => element.dataset.camera === camera[1]);
      throw new Error(`Unsupported document selector: ${selector}`);
    },
    querySelectorAll(selector) {
      assert.equal(selector, "button[data-action]");
      return all.filter((element) => element.tagName === "BUTTON" && element.dataset.action);
    }
  };
}

const cosmosProfiles = [
  {id:"161616", label:"原始 16/16/16", video_steps:16, force_steps:16, action_steps:16},
  {id:"4416", label:"平衡 4/4/16", video_steps:4, force_steps:4, action_steps:16},
  {id:"3310", label:"快速 3/3/10", video_steps:3, force_steps:3, action_steps:10}
];
const jepaProfiles = [
  {id:"dense16", label:"V-JEPA 16/16", video_mode:"single_forward", video_steps:null, force_steps:16, action_steps:16},
  {id:"dense10", label:"V-JEPA 10/10", video_mode:"single_forward", video_steps:null, force_steps:10, action_steps:10}
];

function status(profiles = cosmosProfiles, active = "161616", overrides = {}) {
  const profile = profiles.find((item) => item && item.id === active) || {};
  return {
    mode:"mock", phase:"monitoring", server:{ready:true, url:"offline-stub", backend:"test"},
    sensors:{state:{values:[]}, force:{values:[]}, cameras:[]},
    inference:{enabled:false, actions:[], latencies:[]}, execution:{enabled:false, available:false},
    gripper:{}, events:[],
    sampling:{supported:true, ...profile, sampling_profile:active, profiles, busy:false, error:null, ...overrides}
  };
}

async function harness(initialStatus, controlResponse = {ok:true, message:"offline queued response"}) {
  const document = makeDocument(), timers = new Map(), posts = [], gets = [];
  let response = initialStatus, now = 1000, nextTimer = 1;
  const context = vm.createContext({
    document, performance:{now:() => now}, AbortController, Date, console,
    window:{
      confirm:() => { throw new Error("Blocking confirmation dialogs are forbidden"); },
      alert:() => { throw new Error("Blocking alert dialogs are forbidden"); },
      prompt:() => { throw new Error("Blocking prompt dialogs are forbidden"); },
      addEventListener() {}
    },
    setTimeout(fn, delay) { const id = nextTimer++; timers.set(id, {fn, delay}); return id; },
    clearTimeout(id) { timers.delete(id); }, setInterval() { return nextTimer++; },
    URL:{createObjectURL() { throw new Error("No camera/image IO allowed"); }, revokeObjectURL() {}},
    Image:class { constructor() { throw new Error("No image IO allowed"); } },
    fetch:async (url, options = {}) => {
      if (url === "/api/status") {
        gets.push(url);
        return {ok:true, json:async () => structuredClone(response)};
      }
      assert.equal(url, "/api/control", "Only fake local status/control routes may be used");
      assert.equal(options.method, "POST");
      assert.equal(options.headers["X-Client-Key"], "offline-test-client-key");
      posts.push(JSON.parse(options.body));
      return {ok:controlResponse.ok, json:async () => structuredClone(controlResponse)};
    }
  });
  vm.runInContext(source, context, {filename:path.join(web, "app.js")});
  const flush = async () => { for (let n = 0; n < 10; n++) await Promise.resolve(); };
  await flush();
  function assertRendered() {
    assert.doesNotMatch(document.getElementById("connection-alert").textContent, /异常/, "Full-IIFE rendering must finish without a swallowed DOM error");
    assert.equal(document.getElementById("connection-label").querySelector("span").textContent, "客户端在线");
  }
  assertRendered();
  return {
    document, posts, gets, element:(id) => document.getElementById(id),
    options:() => document.getElementById("sampling-select").options.filter((option) => option.value).map((option) => option.value),
    async choose(value) { const select = document.getElementById("sampling-select"); select.value = value; await select.dispatch("change"); await flush(); },
    async apply() { await document.getElementById("sampling-apply-button").dispatch("click"); await flush(); },
    async click(id) { await document.getElementById(id).dispatch("click"); await flush(); },
    async poll(nextStatus) {
      response = nextStatus;
      const entry = [...timers.entries()].find(([, timer]) => timer.fn.name === "pollStatus");
      assert.ok(entry, "Production pollStatus must schedule the next status request");
      timers.delete(entry[0]); now += 250;
      await entry[1].fn(); await flush(); assertRendered();
    }
  };
}

test("full app replaces Cosmos options with declared V-JEPA options and describes a single visual forward", async () => {
  const app = await harness(status());
  assert.deepEqual(app.options(), ["161616", "4416", "3310"]);
  assert.equal(app.posts.length, 0, "Page initialization must not POST");
  await app.poll(status(jepaProfiles, "dense16"));
  assert.deepEqual(app.options(), ["dense16", "dense10"]);
  assert.equal(app.element("sampling-select").value, "dense16");
  assert.equal(app.element("sampling-steps").textContent, "视觉单次前向；Force/Action 16/16");
  assert.match(app.element("sampling-label").textContent, /视觉单次前向/);
  assert.equal(app.posts.length, 0, "Status updates must not POST");
});

test("polling preserves a valid draft choice and replaces a removed/old-backend choice with the actual profile", async () => {
  const app = await harness(status());
  await app.choose("4416");
  await app.poll(status(cosmosProfiles.map((profile) => ({...profile, label:profile.label + " 已更新"}))));
  assert.equal(app.element("sampling-select").value, "4416");
  assert.equal(app.element("sampling-apply-button").disabled, false);
  await app.poll(status(jepaProfiles, "dense16"));
  assert.equal(app.element("sampling-select").value, "dense16");
  await app.choose("dense10");
  await app.poll(status([jepaProfiles[0]], "dense16"));
  assert.deepEqual(app.options(), ["dense16"]);
  assert.equal(app.element("sampling-select").value, "dense16");
  assert.equal(app.element("sampling-apply-button").disabled, true);
  assert.equal(app.posts.length, 0, "Selecting a profile must not POST");
});

test("only server-declared IDs can POST, including an unfamiliar future profile", async () => {
  const app = await harness(status(jepaProfiles, "dense16"));
  const select = app.element("sampling-select");
  const injected = new Element("option"); injected.value = "4416"; select.append(injected);
  await app.choose("4416");
  assert.equal(app.element("sampling-apply-button").disabled, true);
  await app.apply();
  assert.equal(app.posts.length, 0, "Forged DOM options cannot authorize a profile");
  await app.choose("dense10");
  await app.apply();
  assert.deepEqual(app.posts, [{action:"sampling_set", sampling_profile:"dense10"}]);
  const future = {id:"future-backend-profile", label:"Future profile", video_mode:"single_forward", force_steps:7, action_steps:9};
  await app.poll(status([jepaProfiles[0], future], "dense16"));
  await app.choose(future.id);
  await app.apply();
  assert.deepEqual(app.posts[1], {action:"sampling_set", sampling_profile:future.id});
});

test("missing declarations and unknown actual profile cannot reuse old profile permissions", async () => {
  const app = await harness(status());
  await app.choose("4416");
  await app.poll(status([], "4416"));
  assert.deepEqual(app.options(), []);
  assert.equal(app.element("sampling-select").value, "");
  assert.equal(app.element("sampling-apply-button").disabled, true);
  await app.apply();
  assert.equal(app.posts.length, 0);
  await app.poll(status(jepaProfiles, "unknown-active"));
  assert.equal(app.element("sampling-select").value, "");
  assert.equal(app.element("sampling-apply-button").disabled, true);
});

test("pi05 action-only profiles replace Cosmos controls without implying force denoising", async () => {
  const profiles = [5, 10, 16].map((n) => ({id:`pi05_${n}`, label:`π0.5 · 动作 ${n} 步`, action_steps:n}));
  const app = await harness(status());
  await app.poll(status(profiles, "pi05_10", {model:"pi05", video_mode:"single_forward", force_conditioning:false}));
  assert.equal(app.element("sampling-steps").textContent, "动作 10 步；不使用力条件");
  assert.equal(app.element("sampling-label").textContent, "π0.5 · 动作去噪步数");
  await app.choose("pi05_5");
  await app.apply();
  assert.deepEqual(app.posts, [{action:"sampling_set", sampling_profile:"pi05_5"}]);
  await app.poll(status());
  assert.equal(app.element("sampling-label").textContent, "去噪步数 · 视频 / 力 / 动作");
});

test("declared choices remain blocked while sampling or robot execution is active", async () => {
  const app = await harness(status(jepaProfiles, "dense16"));
  await app.choose("dense10");
  await app.poll(status(jepaProfiles, "dense16", {busy:true}));
  assert.equal(app.element("sampling-apply-button").disabled, true);
  await app.apply();
  assert.equal(app.posts.length, 0);
  const executing = status(jepaProfiles, "dense16"); executing.execution.enabled = true;
  await app.poll(executing);
  assert.equal(app.element("sampling-apply-button").disabled, true);
  await app.apply();
  assert.equal(app.posts.length, 0);
});

test("execute and gripper operations submit once on click without any blocking dialog", async () => {
  const ready = status();
  ready.execution.available = true;
  ready.gripper = {enable_available:true, disable_available:true, motion_available:true,
                   clear_available:true, reconnect_available:true};
  const app = await harness(ready);
  assert.equal(app.posts.length, 0);
  const controls = [
    ["execute-button", "execute_start"], ["gripper-enable-button", "gripper_enable"],
    ["gripper-disable-button", "gripper_disable"], ["gripper-open-button", "gripper_open"],
    ["gripper-close-button", "gripper_close"], ["gripper-clear-button", "gripper_clear_error"],
    ["gripper-reconnect-button", "gripper_reconnect"]
  ];
  for (const [id, action] of controls) {
    const before = app.posts.length;
    assert.equal(app.element(id).disabled, false, `${action} should be available`);
    await app.click(id);
    assert.equal(app.posts.length, before + 1, `${action} should POST exactly once`);
    assert.deepEqual(app.posts.at(-1), {action});
  }
  assert.doesNotMatch(source, /\b(?:window\.)?(?:confirm|alert|prompt)\s*\(/);
});

test("removing dialogs does not bypass existing operation availability guards", async () => {
  const app = await harness(status());
  for (const id of ["execute-button", "gripper-enable-button", "gripper-disable-button", "gripper-open-button", "gripper-close-button", "gripper-clear-button", "gripper-reconnect-button"]) {
    assert.equal(app.element(id).disabled, true);
    await app.click(id);
  }
  assert.equal(app.posts.length, 0);
});

test("pending execution recovery is visible and can be cancelled even when execution is disabled", async () => {
  const waiting = status();
  waiting.recovery = {active:true, reason:"等待新鲜关节反馈", resume_execution:true, attempts:2, last_recovered_at:null};
  const app = await harness(waiting);
  assert.equal(app.element("recovery-notice").hidden, false);
  assert.match(app.element("recovery-notice").textContent, /等待反馈恢复，将自动继续/);
  assert.match(app.element("recovery-notice").textContent, /恢复推理和动作执行/);
  assert.match(app.element("recovery-notice").textContent, /等待新鲜关节反馈/);
  assert.equal(app.element("execution-status").textContent, "等待恢复执行");
  assert.equal(app.element("stop-button").disabled, false);
  assert.equal(app.posts.length, 0, "Recovery display must not issue client-side retry commands");
  await app.click("stop-button");
  assert.deepEqual(app.posts, [{action:"execute_stop"}]);
  const stopped = status(); stopped.recovery = {active:false, resume_execution:false, attempts:2};
  await app.poll(stopped);
  assert.equal(app.element("recovery-notice").hidden, true);
  assert.equal(app.element("stop-button").disabled, true);
});

test("inference-only recovery is described separately and action errors remain in-page", async () => {
  const waiting = status();
  waiting.gripper = {motion_available:true};
  waiting.recovery = {active:true, reason:"相机数据暂未更新", resume_execution:false, attempts:1};
  const app = await harness(waiting, {ok:false, error:"夹爪反馈暂不可用"});
  assert.match(app.element("recovery-notice").textContent, /恢复自动推理，不开启动作执行/);
  assert.equal(app.element("stop-button").disabled, true);
  await app.click("gripper-open-button");
  assert.equal(app.element("command-message").hidden, false);
  assert.match(app.element("command-message").textContent, /操作未完成：夹爪反馈暂不可用/);
  assert.match(app.element("command-message").className, /error/);
});

const savedResetPose = {saved:true, available:true, busy:false, reason:null,
  recorded_at:"2025-01-01T00:00:00+00:00", speed_percent:10,
  joint_degrees:[0, 10, -5, 20, 0, 30, -10], last_result:null};

const timingState = {supported:true, profile:"strict", busy:false, error:null,
  profiles:[{id:"strict", label:"原始时序档", description:"原阈值"},
    {id:"relaxed", label:"宽松实验档", description:"放宽反馈时序"}],
  limits:{gripper_control_max_age_s:.5, feedback_max_age_s:.03,
    camera_max_age_s:.12, camera_pair_skew_s:.05}};

test("timing UI submits only declared profile IDs, preserving a draft through polling", async () => {
  const original = status(); original.timing = structuredClone(timingState);
  const app = await harness(original);
  assert.equal(app.element("timing-panel").hidden, false);
  assert.equal(app.element("timing-select").value, "strict");
  assert.equal(app.element("timing-apply-button").disabled, true);
  assert.match(app.element("timing-limits").textContent, /夹爪动作 500 ms · 关节 30 ms/);
  app.element("timing-select").value = "relaxed";
  await app.element("timing-select").dispatch("change");
  await app.poll(original);
  assert.equal(app.element("timing-select").value, "relaxed");
  assert.equal(app.element("timing-apply-button").disabled, false);
  await app.click("timing-apply-button");
  assert.deepEqual(app.posts, [{action:"timing_set", timing_profile:"relaxed"}]);
  app.element("timing-select").value = "unlimited";
  await app.element("timing-select").dispatch("change");
  await app.click("timing-apply-button");
  assert.equal(app.posts.length, 1);
  const changed = status(); changed.timing = {...timingState, profile:"relaxed"};
  await app.poll(changed);
  assert.equal(app.element("timing-current").textContent, "宽松实验档");
  assert.equal(app.element("timing-select").value, "relaxed");
  assert.equal(app.element("timing-apply-button").disabled, true);
});

test("timing changes are blocked during execution, other operations, or undeclared support", async () => {
  for (const condition of ["execute", "sampling", "hardware", "reconnect", "reset", "timing", "unsupported"]) {
    const current = status(); current.timing = structuredClone(timingState);
    if (condition === "execute") current.execution.enabled = true;
    if (condition === "sampling") current.sampling.busy = true;
    if (condition === "hardware") current.gripper.busy = true;
    if (condition === "reconnect") current.gripper.auto_reconnect = {active:true};
    if (condition === "reset") current.reset_pose = {...savedResetPose, busy:true};
    if (condition === "timing") current.timing.busy = true;
    if (condition === "unsupported") delete current.timing;
    const app = await harness(current);
    app.element("timing-select").value = "relaxed";
    await app.element("timing-select").dispatch("change");
    assert.equal(app.element("timing-select").disabled, true, condition);
    assert.equal(app.element("timing-apply-button").disabled, true, condition);
    await app.click("timing-apply-button");
    assert.equal(app.posts.length, 0, condition);
    if (condition === "unsupported") assert.equal(app.element("timing-panel").hidden, true);
    if (condition === "timing") {
      assert.equal(app.element("execute-button").disabled, true);
      assert.equal(app.element("sampling-select").disabled, true);
      assert.equal(app.element("emergency-button").disabled, false);
    }
  }
});

test("gripper distinguishes usable observation from expired motion feedback", async () => {
  const waiting = status();
  waiting.gripper = {fault_code:0, enabled:true, ready:false, measurement_ready:true,
    age_ms:1200, enabled_age_ms:200, observation_max_age_ms:1500, control_max_age_ms:500,
    samples:12, error:"夹爪运动反馈过期", motion_available:false};
  const app = await harness(waiting);
  assert.equal(app.element("gripper-status").textContent, "观测可用 · 动作等待反馈");
  assert.match(app.element("gripper-cadence").textContent, /推理观测 ≤ 1,500 ms · 动作反馈 ≤ 500 ms/);
  assert.equal(app.element("gripper-open-button").disabled, true);
  waiting.gripper = {...waiting.gripper, age_ms:1600, measurement_ready:false};
  await app.poll(waiting);
  assert.equal(app.element("gripper-status").textContent, "观测反馈过期");
  waiting.gripper = {...waiting.gripper, fault_code:32};
  await app.poll(waiting);
  assert.equal(app.element("gripper-status").textContent, "有报警");
  assert.equal(app.posts.length, 0);
});

test("reset requires a complete saved target and submits once without confirmation while execution is active", async () => {
  const app = await harness(status());
  assert.equal(app.element("reset-pose-button").disabled, true);
  await app.click("reset-pose-button");
  assert.equal(app.posts.length, 0);
  const incomplete = status(); incomplete.reset_pose = {...savedResetPose, joint_degrees:[]};
  await app.poll(incomplete); await app.click("reset-pose-button");
  assert.equal(app.element("reset-pose-button").disabled, true);
  assert.equal(app.posts.length, 0);
  const running = status();
  running.inference.enabled = true; running.execution.enabled = true;
  running.reset_pose = {...savedResetPose};
  await app.poll(running);
  assert.equal(app.element("reset-pose-button").disabled, false);
  assert.equal(app.element("reset-pose-detail").textContent, "10% 速度 · 夹爪保持");
  assert.match(app.element("reset-pose-recorded").textContent, /2025/);
  await app.click("reset-pose-button");
  assert.deepEqual(app.posts, [{action:"reset_pose"}]);
});

test("reset in progress blocks other commands while stop and emergency remain usable", async () => {
  const resetting = status();
  resetting.reset_pose = {...savedResetPose, busy:true};
  resetting.gripper = {enable_available:true, disable_available:true, motion_available:true, clear_available:true, reconnect_available:true};
  resetting.execution = {enabled:false, available:true};
  const app = await harness(resetting);
  await app.choose("4416");
  assert.equal(app.element("reset-pose-button").textContent, "复位中…");
  assert.equal(app.element("execution-status").textContent, "复位中");
  for (const id of ["reset-pose-button", "inference-button", "execute-button", "connect-button", "disconnect-button", "sampling-apply-button", "gripper-enable-button", "gripper-disable-button", "gripper-open-button", "gripper-close-button", "gripper-clear-button", "gripper-reconnect-button"]) {
    assert.equal(app.element(id).disabled, true, `${id} must be disabled during reset`);
    await app.click(id);
  }
  assert.equal(app.element("sampling-select").disabled, true);
  assert.equal(app.posts.length, 0);
  assert.equal(app.element("stop-button").disabled, false);
  assert.equal(app.element("emergency-button").disabled, false);
  await app.click("stop-button"); await app.click("emergency-button");
  assert.deepEqual(app.posts, [{action:"execute_stop"}, {action:"emergency_stop"}]);
  const completed = status(); completed.reset_pose = {...savedResetPose, last_result:{message:"记录位置已到达"}};
  await app.poll(completed);
  assert.equal(app.element("reset-pose-result").textContent, "记录位置已到达");
  assert.equal(app.element("stop-button").disabled, true);
});
