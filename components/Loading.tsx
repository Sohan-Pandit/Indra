
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useEffect, useState, useRef } from 'react';
import { Database, Fingerprint, Activity, Zap } from 'lucide-react';

interface LoadingProps {
  status: string;
  step: number;
  facts?: string[];
}

const Loading: React.FC<LoadingProps> = ({ status, facts = [] }) => {
  const [currentFactIndex, setCurrentFactIndex] = useState(0);
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 });
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (facts.length > 0) {
      const interval = setInterval(() => {
        setCurrentFactIndex((prev) => (prev + 1) % facts.length);
      }, 3500);
      return () => clearInterval(interval);
    }
  }, [facts]);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (containerRef.current) {
      const { left, top, width, height } = containerRef.current.getBoundingClientRect();
      setMousePos({
        x: (e.clientX - left) / width,
        y: (e.clientY - top) / height,
      });
    }
  };

  const parallaxStyle = (factor: number) => ({
    transform: `translate(${(mousePos.x - 0.5) * factor}px, ${(mousePos.y - 0.5) * factor}px)`,
    transition: 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
  });

  return (
    <div 
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className="relative flex flex-col items-center justify-center w-full max-w-4xl mx-auto mt-12 min-h-[500px] overflow-hidden rounded-[3.5rem] bg-white/40 dark:bg-slate-900/20 border border-slate-200/50 dark:border-white/5 backdrop-blur-3xl shadow-2xl transition-all group"
    >
      {/* Interactive Cursor Light */}
      <div 
        className="absolute w-[300px] h-[300px] bg-emerald-500/5 blur-[80px] rounded-full pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-700"
        style={{ 
          left: `${mousePos.x * 100}%`, 
          top: `${mousePos.y * 100}%`,
          transform: 'translate(-50%, -50%)'
        }}
      ></div>

      {/* Central Visual with Parallax */}
      <div className="relative mb-20" style={parallaxStyle(25)}>
        {/* Orbiting Particles reacting to mouse direction */}
        <div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full pointer-events-none"
          style={{ transform: `translate(-50%, -50%) rotate(${(mousePos.x - 0.5) * 20}deg)` }}
        >
          {[0, 1, 2, 3, 4, 5].map((i) => (
             <div 
               key={i} 
               className="absolute" 
               style={{ 
                 animation: `orbit-slow ${8 + i * 2}s linear infinite`, 
                 animationDelay: `-${i * 2}s`,
                 opacity: 0.3 + (i * 0.1)
               }}
             >
                <div className="w-2 h-2 bg-emerald-400 rounded-full blur-[2px] shadow-[0_0_15px_#10b981]"></div>
             </div>
          ))}
        </div>

        <div className="relative z-20 p-10 bg-white dark:bg-slate-800 rounded-[3rem] border border-slate-100 dark:border-white/5 shadow-2xl transition-all duration-700 group-hover:shadow-emerald-500/10 scale-110">
          <Zap className="w-16 h-16 text-emerald-600 dark:text-emerald-400 animate-pulse" />
          <div className="absolute -inset-4 bg-emerald-500/5 blur-3xl rounded-full animate-pulse"></div>
        </div>
      </div>

      {/* Status Info */}
      <div className="relative z-30 w-full max-w-sm px-10 text-center space-y-10" style={parallaxStyle(10)}>
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-2">
            <Zap className="w-4 h-4 text-emerald-500 animate-bounce" />
            <h3 className="text-emerald-700 dark:text-emerald-400 font-black text-[11px] tracking-[0.5em] uppercase leading-none">{status}</h3>
          </div>
          <div className="min-h-[48px] flex items-center justify-center">
            <p className="text-sm text-slate-500 dark:text-slate-400 font-light leading-relaxed italic animate-in fade-in slide-in-from-bottom-2 duration-700">
              {facts[currentFactIndex] || "Synchronizing with Indra core data clusters..."}
            </p>
          </div>
        </div>

        {/* Sophisticated Progress Bar */}
        <div className="relative w-full h-2 bg-slate-100 dark:bg-slate-800/50 rounded-full overflow-hidden border border-slate-200/50 dark:border-white/5">
            <div className="absolute top-0 bottom-0 bg-emerald-500/40 w-1/4 animate-[scan-glow_3s_infinite_ease-in-out]"></div>
            <div className="absolute top-0 bottom-0 bg-emerald-500 w-full animate-pulse opacity-10"></div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-emerald-500/20 to-transparent animate-[shimmer_2s_infinite]"></div>
        </div>

        {/* Capability Tags */}
        <div className="flex justify-center gap-8">
           <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest transition-colors hover:text-emerald-500">
              <Fingerprint className="w-3.5 h-3.5" /> Identity
           </div>
           <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest transition-colors hover:text-emerald-500">
              <Activity className="w-3.5 h-3.5" /> Vectors
           </div>
           <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest transition-colors hover:text-emerald-500">
              <Database className="w-3.5 h-3.5" /> Archive
           </div>
        </div>
      </div>

      <style>{`
        @keyframes orbit-slow {
          from { transform: rotate(0deg) translateX(150px) rotate(0deg); }
          to { transform: rotate(360deg) translateX(150px) rotate(-360deg); }
        }
        @keyframes scan-glow {
          0% { left: -100%; width: 20%; }
          50% { width: 50%; }
          100% { left: 150%; width: 20%; }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
};

export default Loading;
