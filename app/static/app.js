(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const ui = {
    input: $('message'), send: $('send'), messages: $('messages'), welcome: $('welcome'),
    conversation: $('conversation'), error: $('error-message'), newChat: $('new-chat'),
    sidebar: $('sidebar'), scrim: $('mobile-scrim'), menu: $('menu-btn'),
    status: $('model-status'), statusText: $('model-status-text'),
    layout: $('layout'), gate: $('auth-gate'), loginForm: $('login-form'),
    loginId: $('login-id'), loginName: $('login-name'), loginPassword: $('login-password'),
    loginError: $('login-error'), loginSubmit: $('login-submit'),
    sideUser: $('side-user'), sideUserName: $('side-user-name'), sideUserId: $('side-user-id'),
    sideUserAvatar: $('side-user-avatar'), logout: $('logout-btn'),
  };
  const makeId = () => (globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`);
  // 身份来自登录，不再每次随机生成：后端按 student_id 存长期画像、按 session_id
  // 存短期上下文，ID 一变全都找不到 —— 表现就是"它不记得我了"。
  const KEY_TOKEN = 'psy.token';
  const KEY_STUDENT = 'psy.studentId';
  const KEY_NAME = 'psy.name';
  const KEY_SESSION = 'psy.sessionId';
  const readStored = (key) => {
    try { return localStorage.getItem(key) || ''; } catch { return ''; }
  };
  const writeStored = (key, value) => {
    try { localStorage.setItem(key, value); } catch { /* 隐私模式下降级为单次会话 */ }
  };
  const dropStored = (key) => {
    try { localStorage.removeItem(key); } catch { /* 忽略 */ }
  };
  const newId = (key) => {
    const value = makeId();
    writeStored(key, value);
    return value;
  };
  let token = readStored(KEY_TOKEN);
  let studentId = readStored(KEY_STUDENT);
  let displayName = readStored(KEY_NAME);
  let sessionId = readStored(KEY_SESSION) || newId(KEY_SESSION);
  let busy = false;

  function setBusy(value) {
    busy = value;
    ui.input.disabled = value;
    ui.newChat.disabled = value;
    ui.send.disabled = value || !ui.input.value.trim();
    ui.send.setAttribute('aria-label', value ? '正在生成回复' : '发送消息');
  }

  function showError(message) {
    ui.error.textContent = message;
    ui.error.hidden = !message;
  }

  function status(label, type = '') {
    ui.statusText.textContent = label;
    ui.status.className = `status-pill ${type}`;
  }

  async function refreshStatus() {
    try {
      const response = await fetch('/api/meta', { cache: 'no-store' });
      if (!response.ok) throw new Error('meta');
      const data = await response.json();
      const models = data.runtime?.model;
      const main = Array.isArray(models) ? models.find((item) => item.purpose === 'main') : null;
      if (!main) status('服务已连接');
      else if (main.backend === 'mock') status('规则模式', 'fallback');
      else status('服务已连接');
    } catch {
      status('连接异常', 'offline');
    }
  }

  function scrollToLatest() {
    ui.conversation.scrollTop = ui.conversation.scrollHeight;
  }

  function textNode(tag, className, text) {
    const element = document.createElement(tag);
    element.className = className;
    element.textContent = text;
    return element;
  }

  function message(role, text) {
    ui.welcome.hidden = true;
    const row = textNode('div', `message ${role}`, '');
    if (role === 'assistant') {
      const avatar = textNode('div', 'avatar', '');
      avatar.innerHTML = '<svg aria-hidden="true"><use href="#i-spark"></use></svg>';
      row.append(avatar);
    }
    const main = textNode('div', 'message-main', '');
    main.append(textNode('div', 'message-meta', role === 'assistant' ? '心语' : '我'));
    const body = textNode('div', 'message-body', text);
    main.append(body);
    row.append(main);
    ui.messages.append(row);
    scrollToLatest();
    return { row, main, body };
  }

  function note(main, text, type = '') {
    main.append(textNode('div', `reply-note ${type}`, text));
    scrollToLatest();
  }

  function showExercise(main, exercise) {
    if (!exercise || typeof exercise !== 'object') return;
    const card = textNode('section', 'exercise-card', '');
    card.append(textNode('div', 'eyebrow', '给你的一张练习卡'));
    card.append(textNode('h3', '', String(exercise.name || '轻量练习')));
    if (exercise.duration) card.append(textNode('p', '', `建议时长：${exercise.duration}`));
    if (Array.isArray(exercise.steps) && exercise.steps.length) {
      const list = document.createElement('ol');
      for (const step of exercise.steps) list.append(textNode('li', '', String(step)));
      card.append(list);
    }
    main.append(card);
    scrollToLatest();
  }

  // POST 的 SSE 不能用 EventSource；按字节流解码并以空行分帧。
  async function readEvents(response, onEvent) {
    if (!response.body) throw new Error('当前浏览器不支持流式读取');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const handleFrame = (frame) => {
      const lines = frame.split('\n').filter((line) => line.startsWith('data:'));
      if (lines.length) onEvent(JSON.parse(lines.map((line) => line.slice(5).trimStart()).join('\n')));
    };
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, '\n');
        let end;
        while ((end = buffer.indexOf('\n\n')) !== -1) {
          handleFrame(buffer.slice(0, end));
          buffer = buffer.slice(end + 2);
        }
        if (done) {
          if (buffer.trim()) handleFrame(buffer);
          break;
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  async function sendMessage() {
    const text = ui.input.value.trim();
    if (busy || !text) return;
    if (text.length > 4000) {
      showError('内容过长，请控制在 4000 字以内。');
      return;
    }
    showError('');
    setBusy(true);
    message('user', text);
    ui.input.value = '';
    ui.input.style.height = '';
    const reply = message('assistant', '');
    const thinking = document.createElement('span');
    thinking.className = 'thinking';
    thinking.setAttribute('aria-label', '正在思考');
    thinking.innerHTML = '<i></i><i></i><i></i>';
    reply.body.append(thinking);
    let hasDone = false;
    let content = '';
    let cardSeen = false;
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ session_id: sessionId, student_id: studentId, text, speaker: 'student' }),
      });
      if (!response.ok) throw new Error(response.status === 422 ? '输入内容不符合要求。' : `服务请求失败（${response.status}）。`);
      await readEvents(response, (event) => {
        switch (event.type) {
          case 'delta': {
            const chunk = String(event.text || '');
            // 后端练习卡同时发 card 和文字 delta，此处只显示结构化卡，避免重复。
            if (cardSeen && chunk.startsWith('\n\n')) { cardSeen = false; break; }
            cardSeen = false;
            content += chunk;
            reply.body.textContent = content;
            scrollToLatest();
            break;
          }
          case 'card':
            cardSeen = true;
            showExercise(reply.main, event.exercise);
            break;
          case 'blocked':
            content = '';
            reply.body.textContent = '本轮回复已暂停，请立即联系身边可信任的成年人或当地紧急服务。';
            note(reply.main, '系统未放行本轮回复；页面无法确认人工求助是否已送达。请不要等待页面通知，直接寻求帮助。', 'urgent');
            break;
          case 'escalate': {
            const delivery = event.escalation;
            const delivered = delivery?.delivered === true;
            note(reply.main, delivered
              ? '系统记录了风险上报已送达。请同时主动联系身边可信任的成年人，遇到紧急危险请联系当地紧急服务。'
              : '系统未确认风险上报已送达。请立即联系身边可信任的成年人或当地紧急服务，不要等待系统通知。', 'urgent');
            break;
          }
          case 'done':
            hasDone = true;
            if (!content && reply.body.contains(thinking)) {
              reply.body.textContent = '这轮没有可展示的回复。请联系身边可信任的成年人。';
            }
            break;
          default:
            break;
        }
      });
      if (!hasDone) throw new Error('连接提前中断，无法确认本轮是否已处理。请勿直接重复发送相同内容。');
      await refreshStatus();
    } catch (error) {
      if (!content && reply.body.contains(thinking)) reply.body.textContent = '暂时没能取得回复。';
      showError(error.message || '连接发生异常，请检查服务状态。');
      status('连接异常', 'offline');
    } finally {
      setBusy(false);
      ui.input.focus();
      scrollToLatest();
    }
  }

  function closeMenu() {
    ui.sidebar.classList.remove('open');
    ui.scrim.hidden = true;
    ui.menu.setAttribute('aria-expanded', 'false');
  }

  // 刷新页面后把这一会话的历史贴回来 —— 后端仍然记得，界面也要对得上。
  // 短期记忆按学生存，所以这里带上令牌：后端据此确认"这个会话是你的"，
  // 共享电脑上换了账号不会看到上一个人的原话。
  async function restoreHistory() {
    try {
      const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}/history`, {
        cache: 'no-store',
        headers: authHeaders(),
      });
      if (!response.ok) return;
      const data = await response.json();
      for (const item of data.messages || []) {
        const text = String(item.content || '');
        if (!text) continue;
        message(item.role === 'student' ? 'user' : 'assistant', text);
      }
    } catch {
      /* 恢复失败不影响新对话 */
    }
  }

  // ---------------- 登录 ----------------
  function authHeaders(extra) {
    const headers = Object.assign({}, extra);
    if (token) headers.Authorization = `Bearer ${token}`;
    return headers;
  }

  function gateError(text) {
    if (!text) {
      ui.loginError.hidden = true;
      ui.loginError.textContent = '';
      return;
    }
    ui.loginError.textContent = text;
    ui.loginError.hidden = false;
  }

  function showGate(message) {
    ui.gate.hidden = false;
    ui.layout.hidden = true;
    gateError(message || '');
    ui.loginId.focus();
  }

  function renderUser() {
    const label = displayName || '同学';
    ui.sideUser.hidden = false;
    ui.sideUserName.textContent = label;
    ui.sideUserId.textContent = studentId;
    ui.sideUserAvatar.textContent = label.slice(0, 1);
  }

  // 换人（机房共用电脑）时必须换会话：短期记忆按 session 存，
  // 沿用上一个人的 session 会把别人的原话贴到这人屏幕上。
  function switchIfIdentityChanged(nextId) {
    if (studentId && studentId !== nextId) {
      sessionId = newId(KEY_SESSION);
      ui.messages.replaceChildren();
      ui.welcome.hidden = false;
    }
  }

  async function enterApp() {
    ui.gate.hidden = true;
    ui.layout.hidden = false;
    renderUser();
    await restoreHistory();
    refreshStatus();
  }

  async function login(event) {
    event.preventDefault();
    const id = ui.loginId.value.trim();
    const password = ui.loginPassword.value;
    if (id.length < 3) { gateError('学号至少 3 位。'); return; }
    if (password.length < 6) { gateError('密码至少 6 位。'); return; }
    gateError('');
    ui.loginSubmit.disabled = true;
    ui.loginSubmit.textContent = '正在登录…';
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: id, password, name: ui.loginName.value.trim() }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = Array.isArray(data.detail) ? '学号或密码格式不正确。' : data.detail;
        throw new Error(detail || '登录失败，请重试。');
      }
      token = data.token;
      switchIfIdentityChanged(data.student_id);
      studentId = data.student_id;
      displayName = data.name || '';
      writeStored(KEY_TOKEN, token);
      writeStored(KEY_STUDENT, studentId);
      writeStored(KEY_NAME, displayName);
      ui.loginPassword.value = '';
      await enterApp();
    } catch (error) {
      gateError(error.message || '登录失败，请重试。');
    } finally {
      ui.loginSubmit.disabled = false;
      ui.loginSubmit.textContent = '登录';
    }
  }

  async function logout() {
    try {
      await fetch('/api/auth/logout', { method: 'POST', headers: authHeaders() });
    } catch {
      /* 服务不可达也要允许本地退出 */
    }
    token = '';
    studentId = '';
    displayName = '';
    dropStored(KEY_TOKEN);
    dropStored(KEY_STUDENT);
    dropStored(KEY_NAME);
    dropStored(KEY_SESSION);
    sessionId = newId(KEY_SESSION);
    ui.messages.replaceChildren();
    ui.welcome.hidden = false;
    ui.sideUser.hidden = true;
    showGate('');
  }

  // 启动：有令牌先验一次，失效就回登录页；不再"先给人用、背后没身份"。
  async function boot() {
    if (!token) {
      showGate('');
      return;
    }
    try {
      const response = await fetch('/api/auth/me', { headers: authHeaders(), cache: 'no-store' });
      if (!response.ok) throw new Error('stale');
      const me = await response.json();
      switchIfIdentityChanged(me.student_id || studentId);
      studentId = me.student_id || studentId;
      displayName = me.name || displayName;
      writeStored(KEY_STUDENT, studentId);
      writeStored(KEY_NAME, displayName);
      await enterApp();
    } catch {
      token = '';
      dropStored(KEY_TOKEN);
      showGate('登录状态已失效，请重新登录。');
    }
  }

  ui.input.addEventListener('input', () => {
    ui.input.style.height = 'auto';
    ui.input.style.height = `${Math.min(ui.input.scrollHeight, 152)}px`;
    ui.send.disabled = busy || !ui.input.value.trim();
  });
  ui.input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  ui.send.addEventListener('click', sendMessage);
  ui.newChat.addEventListener('click', () => {
    if (busy) return;
    // 只换会话，不换人：换掉 studentId 会让长期画像从零开始
    sessionId = newId(KEY_SESSION);
    ui.messages.replaceChildren();
    ui.welcome.hidden = false;
    showError('');
    closeMenu();
    ui.input.focus();
  });
  for (const button of document.querySelectorAll('[data-prompt]')) {
    button.addEventListener('click', () => {
      ui.input.value = button.dataset.prompt;
      ui.input.dispatchEvent(new Event('input'));
      ui.input.focus();
    });
  }
  ui.menu.addEventListener('click', () => {
    const opened = ui.sidebar.classList.toggle('open');
    ui.scrim.hidden = !opened;
    ui.menu.setAttribute('aria-expanded', String(opened));
  });
  ui.scrim.addEventListener('click', closeMenu);
  ui.loginForm.addEventListener('submit', login);
  ui.logout.addEventListener('click', logout);
  boot();
})();
