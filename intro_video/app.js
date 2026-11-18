// Default Narration Scripts for Major Project Presentation
const DEFAULT_SCRIPTS = [
  "Welcome to the official presentation of our Major Project: Chest X-Ray Diagnostic Report Generator Version 3. An automated end-to-end artificial intelligence radiology system designed for instant clinical report generation.",
  "Radiologists worldwide handle billions of chest X-rays annually, leading to diagnostic backlogs and high workload. Our system eliminates reporting delays by converting medical image scans into clinical text in seconds.",
  "At the core of our V3 architecture is a DenseNet121 visual backbone paired with a Spatial Transformer Encoder. By retaining 7 by 7 spatial feature tokens with 2D positional embeddings, we preserve vital anatomical context.",
  "Our architecture utilizes a multi-task dual-head design. The autoregressive decoder synthesizes full clinical impression text using Beam Search, while the classification head concurrently predicts 14 CheXpert pathologies.",
  "Model training is executed via a 3-phase resumable strategy with optimizer state recovery. We evaluate performance using NLP metrics including BLEU-1 to 4, ROUGE-L, and multi-disease clinical F1 precision scores.",
  "Finally, our unified Streamlit web app offers radiologists a sleek interactive interface with real-time inference, Greedy versus Beam search toggles, and instant clinical report generation. Thank you."
];

// App State
let state = {
  currentSlide: 0,
  isPlaying: false,
  isVoiceEnabled: true,
  isSfxEnabled: true,
  voiceRate: 1.15,
  slideTimer: null,
  slideStartTime: 0,
  slideDuration: 18,
  elapsedSeconds: 0,
  totalSeconds: 118,
  scripts: [...DEFAULT_SCRIPTS],
  isRecording: false,
  mediaRecorder: null,
  recordedChunks: []
};

// Web Audio API Synthesizer Context
let audioCtx = null;

function initAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
}

function playSfx(type) {
  if (!state.isSfxEnabled) return;
  try {
    initAudioContext();
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    const now = audioCtx.currentTime;

    if (type === 'click') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(800, now);
      osc.frequency.exponentialRampToValueAtTime(400, now + 0.05);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);
      osc.start(now);
      osc.stop(now + 0.05);
    } else if (type === 'transition') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(200, now);
      osc.frequency.exponentialRampToValueAtTime(600, now + 0.15);
      gain.gain.setValueAtTime(0.1, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
      osc.start(now);
      osc.stop(now + 0.15);
    } else if (type === 'chime') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, now); // C5
      osc.frequency.setValueAtTime(659.25, now + 0.1); // E5
      osc.frequency.setValueAtTime(783.99, now + 0.2); // G5
      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.4);
      osc.start(now);
      osc.stop(now + 0.4);
    }
  } catch (e) {
    console.warn("Web Audio Sfx error:", e);
  }
}

// Particle Background Canvas
function initParticleBg() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let width, height;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  const particles = [];
  const particleCount = 60;

  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
      color: Math.random() > 0.5 ? 'rgba(0, 240, 255, ' : 'rgba(157, 78, 221, ',
      alpha: Math.random() * 0.5 + 0.2
    });
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);

    // Draw particle connections
    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.15 * (1 - dist / 130)})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Move & draw particles
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color + p.alpha + ')';
      ctx.shadowBlur = 8;
      ctx.shadowColor = 'rgba(0, 240, 255, 0.8)';
      ctx.fill();
    });

    requestAnimationFrame(animate);
  }

  animate();
}

// Speech Synthesis Narration Engine
let speechSynth = window.speechSynthesis;
let currentUtterance = null;

function speakCurrentSlideScript() {
  if (!state.isVoiceEnabled || !speechSynth) return;

  speechSynth.cancel(); // Stop active speech

  const text = state.scripts[state.currentSlide];
  if (!text) return;

  currentUtterance = new SpeechSynthesisUtterance(text);
  currentUtterance.rate = state.voiceRate;
  currentUtterance.pitch = 1.0;

  // Try selecting an English Voice if available
  const voices = speechSynth.getVoices();
  const englishVoice = voices.find(v => v.lang.includes('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('David')));
  if (englishVoice) {
    currentUtterance.voice = englishVoice;
  }

  currentUtterance.onend = function () {
    if (state.isPlaying) {
      // Advance slide shortly after speech finishes if auto playing
      setTimeout(() => {
        if (state.isPlaying) {
          nextSlide();
        }
      }, 800);
    }
  };

  speechSynth.speak(currentUtterance);
}

function stopVoice() {
  if (speechSynth) {
    speechSynth.cancel();
  }
}

// UI & Slide Navigation Logic
const slides = document.querySelectorAll('.slide');
const teleprompterText = document.getElementById('teleprompter-text');
const progressFill = document.getElementById('progress-fill');
const elapsedDisplay = document.getElementById('elapsed-time');
const systemStatus = document.getElementById('system-status');
const playBtnText = document.getElementById('play-btn-text');

function loadSavedScripts() {
  const saved = localStorage.getItem('ai_intro_scripts_v3');
  if (saved) {
    try {
      state.scripts = JSON.parse(saved);
    } catch (e) {
      state.scripts = [...DEFAULT_SCRIPTS];
    }
  }
}

function renderThumbnails() {
  const container = document.getElementById('slide-thumbnails');
  container.innerHTML = '';
  slides.forEach((slide, idx) => {
    const title = slide.querySelector('.slide-title')?.innerText.replace(/\n/g, ' ') || `Scene ${idx + 1}`;
    const thumb = document.createElement('div');
    thumb.className = `thumb-item ${idx === state.currentSlide ? 'active' : ''}`;
    thumb.innerHTML = `
      <span class="thumb-num">0${idx + 1}</span>
      <span class="thumb-title">${title}</span>
    `;
    thumb.addEventListener('click', () => {
      playSfx('click');
      goToSlide(idx);
    });
    container.appendChild(thumb);
  });
}

function goToSlide(index) {
  if (index < 0 || index >= slides.length) return;

  slides[state.currentSlide].classList.remove('active');
  state.currentSlide = index;
  slides[state.currentSlide].classList.add('active');

  // Update Teleprompter
  teleprompterText.innerText = state.scripts[state.currentSlide] || "";

  // Update duration
  state.slideDuration = parseInt(slides[state.currentSlide].getAttribute('data-duration')) || 20;

  renderThumbnails();
  playSfx('transition');

  if (state.isPlaying) {
    startSlideTimer();
    speakCurrentSlideScript();
  } else {
    stopVoice();
  }
}

function nextSlide() {
  if (state.currentSlide < slides.length - 1) {
    goToSlide(state.currentSlide + 1);
  } else {
    // Reached End of Presentation
    pausePresentation();
    goToSlide(0);
    playSfx('chime');
    systemStatus.innerText = "PRESENTATION COMPLETE";
    systemStatus.className = "val text-green";
  }
}

function startSlideTimer() {
  clearInterval(state.slideTimer);
  state.slideStartTime = Date.now();

  state.slideTimer = setInterval(() => {
    if (!state.isPlaying) return;
    const elapsed = (Date.now() - state.slideStartTime) / 1000;
    const percent = Math.min((elapsed / state.slideDuration) * 100, 100);
    progressFill.style.width = `${percent}%`;

    // Update time counter
    state.elapsedSeconds++;
    const mins = String(Math.floor(state.elapsedSeconds / 60)).padStart(2, '0');
    const secs = String(state.elapsedSeconds % 60).padStart(2, '0');
    elapsedDisplay.innerText = `${mins}:${secs}`;
  }, 100);
}

function playPresentation() {
  state.isPlaying = true;
  playBtnText.innerText = "PAUSE VIDEO";
  systemStatus.innerText = "PLAYING SCENE 0" + (state.currentSlide + 1);
  systemStatus.className = "val text-cyan";
  playSfx('chime');
  startSlideTimer();
  speakCurrentSlideScript();
}

function pausePresentation() {
  state.isPlaying = false;
  clearInterval(state.slideTimer);
  playBtnText.innerText = "PLAY VIDEO";
  systemStatus.innerText = "PAUSED";
  systemStatus.className = "val text-green";
  stopVoice();
}

function togglePlayPause() {
  if (state.isPlaying) {
    pausePresentation();
  } else {
    playPresentation();
  }
}

// Screen / Canvas Video Recording Engine
async function startVideoRecording() {
  try {
    const viewport = document.getElementById('screen-viewport');
    
    // Request Screen/Tab Capture
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { displaySurface: "browser", cursor: "never" },
      audio: true
    });

    state.recordedChunks = [];
    const options = { mimeType: 'video/webm;codecs=vp9' };
    state.mediaRecorder = new MediaRecorder(stream, options);

    state.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        state.recordedChunks.push(e.data);
      }
    };

    state.mediaRecorder.onstop = () => {
      const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `AI_Intro_Video_Chest_XRay_V3.webm`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 100);

      document.getElementById('recording-badge').classList.add('hidden');
      document.getElementById('rec-btn-text').innerText = "RECORD VIDEO (.WEBM)";
      state.isRecording = false;
    };

    state.mediaRecorder.start();
    state.isRecording = true;
    document.getElementById('recording-badge').classList.remove('hidden');
    document.getElementById('rec-btn-text').innerText = "STOP RECORDING";

    // Auto start playing presentation from start when recording begins
    goToSlide(0);
    playPresentation();

  } catch (err) {
    console.error("Recording error or cancelled:", err);
    alert("Screen recording was cancelled or not supported by browser.");
  }
}

function stopVideoRecording() {
  if (state.mediaRecorder && state.isRecording) {
    state.mediaRecorder.stop();
  }
}

// Script Modal Editor
function openScriptModal() {
  const container = document.getElementById('script-editor-container');
  container.innerHTML = '';

  state.scripts.forEach((script, idx) => {
    const item = document.createElement('div');
    item.className = 'script-item';
    item.innerHTML = `
      <label>SCENE 0${idx + 1} NARRATION SCRIPT:</label>
      <textarea class="script-textarea" data-idx="${idx}">${script}</textarea>
    `;
    container.appendChild(item);
  });

  document.getElementById('script-modal').classList.remove('hidden');
}

function closeScriptModal() {
  document.getElementById('script-modal').classList.add('hidden');
}

function saveScriptModal() {
  const textareas = document.querySelectorAll('.script-textarea');
  textareas.forEach(ta => {
    const idx = parseInt(ta.getAttribute('data-idx'));
    state.scripts[idx] = ta.value.trim();
  });

  localStorage.setItem('ai_intro_scripts_v3', JSON.stringify(state.scripts));
  closeScriptModal();
  goToSlide(state.currentSlide);
}

// Event Listeners Initialization
function initEventListeners() {
  document.getElementById('btn-play-pause').addEventListener('click', () => {
    playSfx('click');
    togglePlayPause();
  });

  document.getElementById('btn-record').addEventListener('click', () => {
    if (state.isRecording) {
      stopVideoRecording();
    } else {
      startVideoRecording();
    }
  });

  document.getElementById('toggle-voice').addEventListener('click', (e) => {
    state.isVoiceEnabled = !state.isVoiceEnabled;
    e.target.classList.toggle('active', state.isVoiceEnabled);
    e.target.innerText = state.isVoiceEnabled ? 'ON' : 'OFF';
    playSfx('click');
    if (!state.isVoiceEnabled) stopVoice();
  });

  document.getElementById('toggle-sfx').addEventListener('click', (e) => {
    state.isSfxEnabled = !state.isSfxEnabled;
    e.target.classList.toggle('active', state.isSfxEnabled);
    e.target.innerText = state.isSfxEnabled ? 'ON' : 'OFF';
  });

  document.getElementById('voice-rate').addEventListener('change', (e) => {
    state.voiceRate = parseFloat(e.target.value);
    playSfx('click');
    if (state.isPlaying && state.isVoiceEnabled) {
      speakCurrentSlideScript();
    }
  });

  document.getElementById('btn-open-script').addEventListener('click', () => {
    playSfx('click');
    openScriptModal();
  });

  document.getElementById('btn-close-script').addEventListener('click', closeScriptModal);
  document.getElementById('btn-save-script').addEventListener('click', saveScriptModal);

  // Timeline Click Progress
  document.getElementById('progress-bar-container').addEventListener('click', (e) => {
    const rect = e.target.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = clickX / rect.width;
    const targetSlide = Math.floor(ratio * slides.length);
    goToSlide(targetSlide);
  });

  // Keyboard Shortcuts (Space for Play/Pause, Left/Right Arrows for Slides)
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === 'ArrowRight') {
      nextSlide();
    } else if (e.code === 'ArrowLeft') {
      if (state.currentSlide > 0) goToSlide(state.currentSlide - 1);
    }
  });
}

// App Entrypoint
window.addEventListener('DOMContentLoaded', () => {
  loadSavedScripts();
  initParticleBg();
  renderThumbnails();
  initEventListeners();
  goToSlide(0);
});
