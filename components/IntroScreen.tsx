
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useState, useEffect, useRef } from 'react';
import { Play, Fingerprint, Search, ShieldCheck, Zap } from 'lucide-react';

interface IntroScreenProps {
  onComplete: () => void;
}

const IntroScreen: React.FC<IntroScreenProps> = ({ onComplete }) => {
  const [phase, setPhase] = useState(0); 
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer1 = setTimeout(() => setPhase(1), 500); 
    const timer2 = setTimeout(() => setPhase(2), 2500); 
    const timer3 = setTimeout(() => setPhase(3), 4000); 

    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const { left, top, width, height } = containerRef.current.getBoundingClientRect();
        setMousePos({
          x: (e.clientX - left) / width,
          y: (e.clientY - top) / height,
        });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const parallaxStyle = (factor: number) => ({
    transform: `translate(${(mousePos.x - 0.5) * factor}px, ${(mousePos.y - 0.5) * factor}px)`,
    transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
  });

  return (
    <div 
      ref={containerRef}
      className="fixed inset-0 z-[100] bg-[#020617] flex flex-col items-center justify-center overflow-hidden font-display selection:bg-emerald-500/20"
    >
      {/* Background Particles reacting to mouse */}
      <div className="absolute inset-0 opacity-20 pointer-events-none">
        <div 
          className="absolute w-[800px] h-[800px] bg-emerald-500/5 blur-[120px] rounded-full"
          style={{ 
            left: `${mousePos.x * 100}%`, 
            top: `${mousePos.y * 100}%`,
            transform: 'translate(-50%, -50%)',
            transition: 'all 1s cubic-bezier(0.16, 1, 0.3, 1)'
          }}
        ></div>
      </div>

      <div className="relative flex flex-col items-center gap-12 max-w-lg w-full px-8">
        
        {/* Cinematic Core visual with Parallax */}
        <div 
          className={`relative w-64 h-64 flex items-center justify-center transition-all duration-1000 ${phase >= 2 ? 'scale-90' : 'scale-100'}`}
          style={parallaxStyle(40)}
        >
          {/* Animated Rings */}
          <div className={`absolute inset-0 border border-emerald-500/10 rounded-[4rem] animate-[spin_20s_linear_infinite] ${phase >= 1 ? 'opacity-100' : 'opacity-0'}`}></div>
          <div className={`absolute inset-6 border border-teal-500/10 rounded-[3.5rem] animate-[spin_15s_linear_infinite_reverse] ${phase >= 1 ? 'opacity-100' : 'opacity-0'}`}></div>
          <div className={`absolute inset-12 border border-emerald-500/5 rounded-[3rem] animate-[spin_10s_linear_infinite] ${phase >= 1 ? 'opacity-100' : 'opacity-0'}`}></div>
          
          {/* Central Logo Box */}
          <div 
            className={`relative z-10 p-10 bg-slate-900 rounded-[3rem] border border-white/5 shadow-2xl transition-all duration-1000 ${phase >= 2 ? 'bg-emerald-600 border-emerald-400 shadow-emerald-500/40' : ''}`}
            style={parallaxStyle(15)}
          >
             <Zap className={`w-16 h-16 transition-all duration-1000 ${phase >= 2 ? 'text-white' : 'text-emerald-500'}`} />
             {phase >= 2 && (
               <div className="absolute -inset-2 bg-emerald-500/20 blur-xl rounded-full animate-pulse"></div>
             )}
          </div>
        </div>

        {/* Branding with Reveal Animation */}
        <div className={`flex flex-col items-center gap-2 transition-all duration-1000 ${phase >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
           <h1 className="text-7xl font-display font-bold text-white tracking-tighter relative group">
              Indra
              <span className="absolute -right-8 top-0 text-emerald-500 text-sm animate-pulse">
                <Zap className="w-4 h-4" />
              </span>
           </h1>
           <div className={`h-[2px] w-24 bg-gradient-to-r from-transparent via-emerald-500 to-transparent transition-all duration-1000 ${phase >= 1 ? 'scale-x-100' : 'scale-x-0'}`}></div>
           <p className="text-[12px] font-black text-emerald-500/60 uppercase tracking-[0.5em] mt-2">
              Scientific Intelligence
           </p>
        </div>

        {/* Intuitive Action Area */}
        <div className={`w-full space-y-8 transition-all duration-1000 delay-700 ${phase >= 3 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'}`}>
           <button 
              onClick={onComplete}
              className="group relative w-full py-6 rounded-[2rem] overflow-hidden transition-all hover:scale-[1.02] active:scale-[0.98] shadow-2xl shadow-emerald-950/20"
           >
              <div className="absolute inset-0 bg-emerald-600 group-hover:bg-emerald-500 transition-colors"></div>
              <div className="relative flex items-center justify-center gap-4 text-white font-bold tracking-widest text-sm">
                <span>INITIALIZE NEURAL INTERFACE</span>
                <Play className="w-4 h-4 fill-current group-hover:translate-x-1 transition-transform" />
              </div>
              <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
           </button>
           
           {/* Capability Indicators */}
           <div className="flex justify-between px-4">
              {[
                { icon: Fingerprint, label: 'Verified', desc: 'Grounding' },
                { icon: Search, label: 'Semantic', desc: 'Detection' },
                { icon: ShieldCheck, label: 'Research', desc: 'Secure' }
              ].map((item, i) => (
                <div key={i} className="flex flex-col items-center gap-2 group cursor-help">
                  <div className="p-3 rounded-2xl bg-white/5 border border-white/5 group-hover:bg-emerald-500/10 group-hover:border-emerald-500/20 transition-all">
                    <item.icon className="w-5 h-5 text-white/40 group-hover:text-emerald-500" />
                  </div>
                  <div className="text-center">
                    <p className="text-[9px] font-black uppercase tracking-widest text-white/30 group-hover:text-white/60">{item.label}</p>
                    <p className="text-[8px] font-medium text-white/10">{item.desc}</p>
                  </div>
                </div>
              ))}
           </div>
        </div>

      </div>

      {/* Dynamic Grid reacting slightly */}
      <div 
        className="absolute inset-0 opacity-[0.05] pointer-events-none" 
        style={{ 
          backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)', 
          backgroundSize: '80px 80px',
          ...parallaxStyle(-20)
        }}
      ></div>

      <button 
        onClick={onComplete}
        className="absolute bottom-12 text-[10px] font-black text-slate-700 hover:text-emerald-500 transition-all uppercase tracking-[0.4em] px-6 py-3 border border-transparent hover:border-emerald-500/20 rounded-full"
      >
        Bypass Handshake
      </button>
    </div>
  );
};

export default IntroScreen;
