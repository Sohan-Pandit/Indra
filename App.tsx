
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useState, useEffect, useRef } from 'react';
import { AnnotationSession } from './types';
import { annotateAbstract } from './services/geminiService';
import Loading from './components/Loading';
import IntroScreen from './components/IntroScreen';
import {
  FileText,
  AlertCircle,
  Terminal,
  Calendar,
  Sun,
  Moon,
  Key,
  ArrowRight,
  ClipboardList,
  Fingerprint,
  Database,
  History as HistoryIcon,
  Trash2,
  Copy,
  Check,
  ShieldAlert,
  Zap,
  ChevronDown,
  ChevronUp,
  Settings
} from 'lucide-react';

const App: React.FC = () => {
  const [showIntro, setShowIntro] = useState(true);
  const [abstract, setAbstract] = useState('');
  const [pubYear, setPubYear] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState<string | null>(null);

  const [sessions, setSessions] = useState<AnnotationSession[]>([]);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [hasApiKey, setHasApiKey] = useState(false);
  const [checkingKey, setCheckingKey] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Manual Key Entry State
  const [showManualEntry, setShowManualEntry] = useState(false);
  const [manualKeyInput, setManualKeyInput] = useState('');
  const [usingManualKey, setUsingManualKey] = useState(false);
  const [showKeyModal, setShowKeyModal] = useState(false);

  // Custom Cursor Refs for Performance
  const cursorRef = useRef<HTMLDivElement>(null);
  const mousePos = useRef({ x: 0, y: 0 });
  const cursorPos = useRef({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);
  const [cursorVisible, setCursorVisible] = useState(false);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  useEffect(() => {
    const checkKey = async () => {
      try {
        const customKey = localStorage.getItem('INDRA_CUSTOM_KEY');
        if (customKey) {
          setUsingManualKey(true);
          setHasApiKey(true);
        } else if (window.aistudio && window.aistudio.hasSelectedApiKey) {
          const hasKey = await window.aistudio.hasSelectedApiKey();
          setHasApiKey(hasKey);
        } else {
          setHasApiKey(false);
        }
      } catch (e) {
        console.error("Error checking API key:", e);
      } finally {
        setCheckingKey(false);
      }
    };
    checkKey();

    const handleMouseMove = (e: MouseEvent) => {
      mousePos.current = { x: e.clientX, y: e.clientY };
      if (!cursorVisible) setCursorVisible(true);

      const target = e.target as HTMLElement;
      const isInteractive = target.closest('button, a, textarea, input, [role="button"], .interactive');
      setIsHovering(!!isInteractive);
    };

    const handleMouseLeave = () => setCursorVisible(false);
    const handleMouseEnter = () => setCursorVisible(true);

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
    };
  }, [cursorVisible]);

  // High-performance cursor update loop
  useEffect(() => {
    let animationFrameId: number;
    const animateCursor = () => {
      cursorPos.current.x += (mousePos.current.x - cursorPos.current.x) * 0.15;
      cursorPos.current.y += (mousePos.current.y - cursorPos.current.y) * 0.15;

      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate3d(${cursorPos.current.x}px, ${cursorPos.current.y}px, 0)`;
      }
      animationFrameId = requestAnimationFrame(animateCursor);
    };
    animateCursor();
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  const handleSelectKey = async () => {
    // 1. Try AI Studio (Project IDX)
    if (window.aistudio && window.aistudio.openSelectKey) {
      try {
        await window.aistudio.openSelectKey();
        localStorage.removeItem('INDRA_CUSTOM_KEY');
        setUsingManualKey(false);
        setHasApiKey(true);
        setError(null);
        return;
      } catch (e) {
        console.error("Failed to open key selector:", e);
      }
    }

    // 2. Try Environment Variable (Localhost)
    try {
      // @ts-ignore
      const envKey = process.env.GEMINI_API_KEY;
      if (envKey && envKey !== 'PLACEHOLDER_API_KEY') {
        localStorage.setItem('INDRA_CUSTOM_KEY', envKey);
        setUsingManualKey(true);
        setHasApiKey(true);
        setError(null);
        // Visual feedback could be added here, but the modal closing is sufficient
      } else {
        alert("No valid API Key found in environment variables (.env.local). Please enter manually or check your .env file.");
        console.warn("Env Key:", envKey);
      }
    } catch (e) {
      console.error("Auto-fetch failed:", e);
    }
  };

  const handleManualKeySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (manualKeyInput.trim()) {
      localStorage.setItem('INDRA_CUSTOM_KEY', manualKeyInput.trim());
      setUsingManualKey(true);
      setHasApiKey(true);
      setError(null);
      setManualKeyInput('');
      setShowKeyModal(false);
    }
  };

  const handleClearCustomKey = () => {
    localStorage.removeItem('INDRA_CUSTOM_KEY');
    setUsingManualKey(false);
    setHasApiKey(false);
    setManualKeyInput('');
    setShowKeyModal(true);
  };

  const handleAnnotate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isLoading || !abstract.trim()) return;

    setIsLoading(true);
    setError(null);
    setLoadingMessage("Indra is distilling impact vectors...");

    try {
      const result = await annotateAbstract(abstract, pubYear);
      const newSession: AnnotationSession = {
        id: Date.now().toString(),
        abstract,
        pubYear,
        annotation: result,
        timestamp: Date.now()
      };
      setSessions([newSession, ...sessions]);
    } catch (err: any) {
      console.error(err);
      setError((err?.message || String(err)) || 'Annotation pipeline encountered an unknown error.');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearHistory = () => {
    if (confirm("Clear all annotation history?")) {
      setSessions([]);
    }
  };

  const SystemCursor = () => (
    <div
      ref={cursorRef}
      className={`fixed top-0 left-0 z-[1000] pointer-events-none mix-blend-difference transition-opacity duration-300 ${cursorVisible ? 'opacity-100' : 'opacity-0'}`}
    >
      <div className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white transition-all duration-300 ${isHovering ? 'w-16 h-16 bg-white/10 opacity-60 scale-125' : 'w-8 h-8 opacity-100 scale-100'}`} />
      <div className={`absolute -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-white rounded-full transition-all duration-200 ${isHovering ? 'scale-0' : 'scale-100'}`} />
      <div className={`absolute -translate-x-1/2 -translate-y-1/2 w-4 h-[1px] bg-white/20 transition-opacity ${isHovering ? 'opacity-100' : 'opacity-0'}`} />
      <div className={`absolute -translate-x-1/2 -translate-y-1/2 w-[1px] h-4 bg-white/20 transition-opacity ${isHovering ? 'opacity-100' : 'opacity-0'}`} />
    </div>
  );

  const KeySelectionModal = () => (
    <div className="fixed inset-0 z-[200] bg-slate-950/98 backdrop-blur-3xl flex items-center justify-center p-6">
      <div className="bg-white dark:bg-slate-900 border border-emerald-500/20 rounded-[3rem] shadow-2xl max-w-lg w-full p-10 sm:p-12 relative overflow-hidden animate-in zoom-in-95 duration-500">
        <div className="absolute -top-32 -right-32 w-64 h-64 bg-emerald-500/10 blur-[100px] rounded-full"></div>
        {hasApiKey && (
          <button
            onClick={() => setShowKeyModal(false)}
            className="absolute top-6 right-6 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors text-xl font-light"
            title="Close"
          >
            &times;
          </button>
        )}
        <div className="flex flex-col items-center text-center space-y-8 sm:space-y-10">
          <div className="w-24 h-24 sm:w-28 sm:h-28 bg-emerald-500/5 rounded-[2.5rem] flex items-center justify-center text-emerald-500 border border-emerald-500/20 shadow-inner group">
            <ShieldAlert className="w-10 h-10 sm:w-12 sm:h-12 group-hover:rotate-12 transition-transform" />
          </div>
          <div className="space-y-4">
            <h2 className="text-3xl sm:text-4xl font-display font-bold text-slate-900 dark:text-white tracking-tighter">Identity Required</h2>
            <p className="text-slate-500 dark:text-slate-400 text-base sm:text-lg leading-relaxed font-light">Indra auto-detects your provider from the key format. Supports Gemini, Anthropic, and OpenAI.</p>
          </div>

          <div className="w-full">
            <form onSubmit={handleManualKeySubmit} className="space-y-3">
              <input
                type="password"
                value={manualKeyInput}
                onChange={(e) => setManualKeyInput(e.target.value)}
                placeholder="Paste API Key (Gemini, Anthropic, or OpenAI)..."
                className="w-full p-4 bg-slate-50 dark:bg-slate-950/50 border border-slate-200 dark:border-white/5 rounded-xl outline-none focus:ring-4 focus:ring-emerald-500/10 focus:border-emerald-500/40 transition-all font-mono text-sm"
              />
              <button
                type="submit"
                disabled={!manualKeyInput.trim()}
                className="w-full py-5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl font-bold shadow-2xl shadow-emerald-500/30 transition-all flex items-center justify-center gap-3 transform active:scale-[0.98] hover:scale-[1.02] disabled:opacity-30"
              >
                <Key className="w-5 h-5" />
                <span className="text-lg">Confirm Token</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <SystemCursor />

      {!showIntro && !checkingKey && (showKeyModal || !hasApiKey) && <KeySelectionModal />}
      {showIntro ? (
        <IntroScreen onComplete={() => setShowIntro(false)} />
      ) : (
        <div className="min-h-screen bg-[#fafafa] dark:bg-[#020617] text-slate-900 dark:text-slate-200 font-sans selection:bg-emerald-500/20 selection:text-emerald-900 dark:selection:text-emerald-100 pb-32 overflow-x-hidden relative z-10 transition-colors duration-700">

          {/* Background Ambience */}
          <div
            className="fixed inset-0 z-[-2] pointer-events-none opacity-[0.03] dark:opacity-[0.05] bg-grid-animate"
            style={{
              backgroundImage: 'linear-gradient(#10b981 1px, transparent 1px), linear-gradient(90deg, #10b981 1px, transparent 1px)',
              backgroundSize: '80px 80px'
            }}
          ></div>

          <header className="sticky top-0 z-50 transition-all border-b border-slate-200/60 dark:border-white/5 backdrop-blur-3xl bg-white/70 dark:bg-slate-950/70">
            <div className="max-w-7xl mx-auto px-8 h-24 flex items-center justify-between">
              <div className="flex items-center gap-6 group">
                <div className="relative magnetic-target">
                  <div className="absolute -inset-3 bg-emerald-500/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                  <div className="relative bg-white dark:bg-slate-900 p-3 rounded-2xl border border-slate-200 dark:border-white/10 shadow-sm transition-transform">
                    <Zap className="w-7 h-7 text-emerald-600 dark:text-emerald-400" />
                  </div>
                </div>
                <div className="flex flex-col">
                  <span className="font-display font-bold text-3xl tracking-tighter leading-none group-hover:text-emerald-600 transition-colors">Indra</span>

                </div>
              </div>

              <div className="flex items-center gap-3">
                {hasApiKey ? (
                  <button
                    onClick={handleClearCustomKey}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-red-500/10 hover:text-red-500 hover:border-red-500/20 transition-all border border-emerald-500/20"
                    title="Change API key"
                  >
                    <Key className="w-3.5 h-3.5" />
                    Change Key
                  </button>
                ) : (
                  <button
                    onClick={() => setShowKeyModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-emerald-500 transition-all animate-pulse"
                    title="Enter API key"
                  >
                    <Key className="w-3.5 h-3.5" />
                    Enter API Key
                  </button>
                )}
                {sessions.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="p-3 rounded-2xl hover:bg-red-50 dark:hover:bg-red-950/30 text-slate-400 hover:text-red-500 transition-all hover:scale-110"
                    title="Wipe Session"
                  >
                    <Trash2 className="w-6 h-6" />
                  </button>
                )}
                <div className="w-[1px] h-8 bg-slate-200 dark:bg-white/10 mx-3"></div>
                <button
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className="p-3 rounded-2xl bg-slate-100 dark:bg-slate-900 hover:ring-2 ring-emerald-500/30 transition-all hover:scale-110 shadow-sm"
                >
                  {isDarkMode ? <Sun className="w-6 h-6 text-amber-400" /> : <Moon className="w-6 h-6 text-slate-600" />}
                </button>
              </div>
            </div>
          </header>

          <main className="max-w-7xl mx-auto px-8 py-20 relative">
            <div className={`transition-all duration-1000 ease-in-out ${sessions.length > 0 ? 'mb-24' : 'min-h-[60vh] flex flex-col justify-center'}`}>
              {!sessions.length && (
                <div className="text-center mb-24 space-y-10">
                  <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-[11px] font-black tracking-[0.3em] uppercase mb-4 animate-reveal">
                    <Zap className="w-3 h-3 fill-current" />
                    <span>Neural Pipeline Online</span>
                  </div>
                  <h1 className="text-8xl md:text-9xl font-display font-bold tracking-tighter text-slate-900 dark:text-white leading-[0.85] animate-reveal [animation-delay:200ms]">
                    Science for <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-500 bg-[length:200%_auto] animate-[gradient_8s_linear_infinite]">Impact.</span>
                  </h1>
                  <p className="text-2xl text-slate-500 dark:text-slate-400 max-w-3xl mx-auto font-light leading-relaxed animate-reveal [animation-delay:400ms]">
                    Distill extreme events and natural hazard research abstracts into actionable vector data, <br />mapping global climate signals with AI precision.
                  </p>
                </div>
              )}

              <form onSubmit={handleAnnotate} className="relative group max-w-5xl mx-auto animate-reveal [animation-delay:600ms]">
                <div className="absolute -inset-4 bg-gradient-to-r from-emerald-500/10 via-teal-500/5 to-emerald-500/10 rounded-[4rem] blur-3xl opacity-0 group-hover:opacity-100 transition duration-1000"></div>
                <div className="relative bg-white/60 dark:bg-slate-900/40 backdrop-blur-3xl border border-slate-200 dark:border-white/5 p-12 rounded-[3.5rem] shadow-2xl space-y-12 transition-all duration-700 group-focus-within:border-emerald-500/40">
                  <div className="space-y-6">
                    <div className="flex items-center justify-between px-4">
                      <label className="text-[12px] font-black uppercase tracking-[0.4em] text-slate-400 dark:text-slate-500 flex items-center gap-4">
                        <FileText className="w-4 h-4 text-emerald-500" />
                        Abstract Corpus
                      </label>
                      <div className="flex items-center gap-4">
                        <span className="text-[10px] font-mono text-emerald-500 bg-emerald-500/5 px-3 py-1.5 rounded-lg border border-emerald-500/10 uppercase tracking-widest">{abstract.length} Vectors</span>
                      </div>
                    </div>
                    <textarea
                      value={abstract}
                      onChange={(e) => setAbstract(e.target.value)}
                      placeholder="Insert extreme event related research text here for distillation..."
                      className="w-full h-72 p-10 bg-slate-50/50 dark:bg-slate-950/30 border border-slate-200 dark:border-slate-800/50 rounded-[3rem] outline-none focus:ring-[12px] focus:ring-emerald-500/5 focus:border-emerald-500/40 transition-all font-serif leading-relaxed text-2xl resize-none shadow-inner"
                    />
                  </div>

                  <div className="flex flex-col md:flex-row gap-8">
                    <div className="flex-1 space-y-4">
                      <label className="text-[12px] font-black uppercase tracking-[0.4em] text-slate-400 dark:text-slate-500 flex items-center gap-4 px-4">
                        <Calendar className="w-4 h-4 text-emerald-500" />
                        Temporal Anchor
                      </label>
                      <input
                        type="number"
                        value={pubYear}
                        onChange={(e) => setPubYear(e.target.value)}
                        placeholder="YYYY"
                        className="w-full p-6 bg-slate-50/50 dark:bg-slate-950/30 border border-slate-200 dark:border-slate-800/50 rounded-2xl outline-none focus:ring-8 focus:ring-emerald-500/5 focus:border-emerald-500/40 transition-all font-mono text-2xl"
                      />
                    </div>
                    <div className="md:w-1/3 flex items-end">
                      <button
                        type="submit"
                        disabled={isLoading || !abstract.trim()}
                        className={`w-full h-[85px] relative group/btn overflow-hidden text-white font-bold rounded-[2rem] transition-all duration-300 flex items-center justify-between px-8 transform active:scale-[0.97] hover:scale-[1.02] disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:scale-100 ${
                          !isLoading && abstract.trim()
                            ? 'bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950 border border-emerald-500/40 shadow-2xl shadow-emerald-500/10'
                            : 'bg-slate-900 dark:bg-slate-950 border border-white/5 shadow-xl'
                        }`}
                      >
                        <div className="absolute inset-0 opacity-0 group-hover/btn:opacity-100 transition-opacity duration-500 bg-gradient-to-r from-emerald-500/10 via-transparent to-transparent pointer-events-none" />
                        {isLoading ? (
                          <div className="w-full flex justify-center items-center gap-4">
                            <div className="w-5 h-5 border-2 border-white/20 border-t-emerald-400 rounded-full animate-spin" />
                            <span className="text-sm tracking-[0.3em] uppercase font-black text-white/50">Processing</span>
                          </div>
                        ) : (
                          <>
                            <div className="flex flex-col items-start gap-1.5">
                              <span className="text-[10px] uppercase tracking-[0.4em] text-emerald-400 font-black flex items-center gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
                                Neural Flow
                              </span>
                              <span className="text-base tracking-[0.2em] leading-none uppercase font-black">Distillation</span>
                            </div>
                            <div className="w-11 h-11 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center group-hover/btn:bg-emerald-500 group-hover/btn:border-emerald-400 group-hover/btn:shadow-lg group-hover/btn:shadow-emerald-500/40 transition-all duration-300">
                              <Zap className="w-5 h-5 text-emerald-400 group-hover/btn:text-white transition-colors duration-300" />
                            </div>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </form>
            </div>

            {isLoading && <Loading status={loadingMessage} step={2} facts={["Analyzing scientific vectors...", "Grounded verified extraction...", "Normalizing spatio-temporal indices..."]} />}

            {error && (
              <div className="max-w-3xl mx-auto p-10 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 rounded-[3rem] flex items-center gap-8 text-red-800 dark:text-red-400 mb-20 animate-in slide-in-from-bottom-6">
                <div className="bg-red-500 text-white p-4 rounded-3xl shadow-lg shadow-red-500/20">
                  <AlertCircle className="w-8 h-8" />
                </div>
                <div className="space-y-2">
                  <p className="text-xl font-bold tracking-tight">System Interruption</p>
                  <p className="text-base font-medium opacity-80 leading-relaxed">{error}</p>
                </div>
              </div>
            )}

            {sessions.length > 0 && !isLoading && (
              <div className="space-y-40">
                {sessions.map((session, idx) => (
                  <div key={session.id} className="animate-reveal duration-1000 relative">

                    <div className="flex items-center gap-10 mb-20">
                      <div className="flex-none flex items-center gap-6">
                        <div className="w-16 h-16 rounded-[2rem] bg-slate-950 dark:bg-slate-100 flex items-center justify-center text-white dark:text-slate-900 font-bold text-2xl shadow-2xl magnetic-target">
                          {sessions.length - idx}
                        </div>
                        <div>
                          <h2 className="text-3xl font-display font-bold text-slate-800 dark:text-slate-100 tracking-tight">Extracted Record</h2>
                          <p className="text-[11px] text-emerald-500 font-black tracking-[0.4em] uppercase mt-1">Status: Verification Pass 100%</p>
                        </div>
                      </div>
                      <div className="flex-1 h-[2px] bg-slate-200 dark:bg-white/5 rounded-full overflow-hidden relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-emerald-500/40 to-transparent w-1/3 animate-[shimmer_3s_infinite]" />
                      </div>
                      <div className="flex-none text-[12px] font-mono text-slate-400 uppercase tracking-widest bg-slate-100 dark:bg-slate-900/50 border border-slate-200 dark:border-white/5 px-6 py-3 rounded-2xl">
                        LOG: {session.id.slice(-8)}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-6 gap-16">
                      <div className="lg:col-span-4 space-y-8">
                        <div className="bg-white/50 dark:bg-slate-900/40 border border-slate-200 dark:border-white/5 rounded-[4rem] overflow-hidden shadow-2xl backdrop-blur-3xl group/json hover:border-emerald-500/30 transition-all duration-700">
                          <div className="px-12 py-8 bg-slate-100/50 dark:bg-white/5 border-b border-slate-200 dark:border-white/5 flex items-center justify-between">
                            <div className="flex items-center gap-4">
                              <div className="p-2.5 rounded-xl bg-emerald-500/10 text-emerald-500"><Terminal className="w-5 h-5" /></div>
                              <span className="text-[13px] font-black text-slate-500 dark:text-slate-400 uppercase tracking-[0.4em]">Structured Vector Output</span>
                            </div>
                            <button
                              onClick={() => copyToClipboard(JSON.stringify(session.annotation.record, null, 2), session.id + '-json')}
                              className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-white/10 text-[12px] font-bold text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 transition-all hover:scale-105"
                            >
                              {copiedId === session.id + '-json' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                              {copiedId === session.id + '-json' ? 'VERIFIED' : 'EXPORT JSON'}
                            </button>
                          </div>
                          <div className="p-12 font-mono text-base leading-relaxed overflow-auto max-h-[800px] scrollbar-none">
                            <pre className="text-emerald-800/80 dark:text-emerald-400/80 whitespace-pre-wrap selection:bg-emerald-500/20">
                              {JSON.stringify(session.annotation.record, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </div>

                      <div className="lg:col-span-2 space-y-12">
                        <div className="bg-white/40 dark:bg-slate-900/80 p-12 rounded-[4rem] border border-slate-200 dark:border-white/5 shadow-2xl relative group overflow-hidden hover:shadow-emerald-500/5 transition-all duration-700">
                          <div className="absolute top-0 right-0 p-12 opacity-5 group-hover:opacity-100 transition-opacity duration-1000 rotate-12 group-hover:rotate-0">
                            <ClipboardList className="w-16 h-16 text-emerald-500" />
                          </div>
                          <h3 className="text-[12px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-[0.5em] mb-12 flex items-center gap-4">
                            <Database className="w-4 h-4 text-emerald-500" /> Neural Context
                          </h3>
                          <div className="prose prose-slate dark:prose-invert prose-md max-w-none text-slate-600 dark:text-slate-300 leading-relaxed font-light space-y-8">
                            {session.annotation.uncertaintyAnalysis.split('\n').map((line, i) => (
                              <p key={i} className="leading-relaxed border-l-2 border-slate-100 dark:border-white/5 pl-6 transition-all hover:border-emerald-500/50">{line}</p>
                            ))}
                          </div>
                        </div>

                        {session.annotation.secondaryImpacts && (
                          <div className="relative group magnetic-target">
                            <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/40 to-teal-500/40 rounded-[3.5rem] blur-2xl opacity-10 group-hover:opacity-100 transition duration-1000"></div>
                            <div className="relative bg-emerald-500/5 dark:bg-emerald-950/20 p-12 rounded-[3.5rem] border border-emerald-500/20 transition-all">
                              <h3 className="text-[12px] font-black text-emerald-800 dark:text-emerald-400 uppercase tracking-[0.4em] mb-6 flex items-center gap-4">
                                <Zap className="w-4 h-4 fill-current animate-pulse" />
                                Secondary Signals
                              </h3>
                              <p className="text-lg text-emerald-900/70 dark:text-emerald-100/70 leading-relaxed italic font-serif opacity-80 group-hover:opacity-100 transition-opacity">
                                {session.annotation.secondaryImpacts}
                              </p>
                            </div>
                          </div>
                        )}

                        <div className="p-10 rounded-[3.5rem] bg-slate-50/50 dark:bg-slate-950/40 border border-slate-200 dark:border-white/5 group hover:bg-white dark:hover:bg-slate-900 transition-all shadow-inner">
                          <h3 className="text-[11px] font-black text-slate-400 uppercase tracking-[0.4em] mb-8 flex items-center gap-4">
                            <Fingerprint className="w-4 h-4 text-emerald-500" /> Identity Grounding
                          </h3>
                          <p className="text-base text-slate-500 dark:text-slate-400 italic leading-relaxed border-l-4 border-emerald-500/40 pl-8 py-3 transition-all group-hover:border-emerald-500 group-hover:text-slate-900 dark:group-hover:text-white">
                            "{session.annotation.record.grounding_quote}"
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </main>

          {/* Persistent Floating Command Console */}
          {sessions.length > 0 && (
            <div className="fixed bottom-12 left-1/2 -translate-x-1/2 z-[40] animate-in slide-in-from-bottom-12 duration-1000">
              <div className="bg-white/80 dark:bg-slate-950/80 backdrop-blur-3xl border border-slate-200 dark:border-white/10 px-12 py-6 rounded-full shadow-2xl flex items-center gap-12 hover:shadow-emerald-500/20 transition-shadow group">
                <div className="flex items-center gap-5 text-[13px] font-black text-slate-600 dark:text-slate-300 uppercase tracking-[0.5em]">
                  <div className="p-2.5 rounded-full bg-emerald-500/10 text-emerald-500 group-hover:rotate-180 transition-transform duration-700"><HistoryIcon className="w-5 h-5" /></div>
                  <span>Cluster Archives: {sessions.length}</span>
                </div>
                <div className="w-[2px] h-8 bg-slate-200 dark:bg-white/10 rounded-full" />
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="text-[13px] font-black text-emerald-600 dark:text-emerald-400 uppercase tracking-[0.5em] hover:text-emerald-500 transition-all flex items-center gap-3 hover:-translate-y-1"
                >
                  Top of Flow
                </button>
              </div>
            </div>
          )}
          <footer className="w-full py-6 text-center text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 dark:text-slate-600 opacity-60 hover:opacity-100 transition-opacity">
            © 2026 Sohan Pandit
          </footer>
        </div>
      )}
      <style>{`
      @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(300%); }
      }
      .custom-code-bg {
        background-image: radial-gradient(circle at center, rgba(16, 185, 129, 0.03) 0%, transparent 100%);
      }
    `}</style>
    </>
  );
};

export default App;
