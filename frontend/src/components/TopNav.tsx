import React from 'react'
import { RotateCcw, Cpu, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react'

interface TopNavProps {
  isConnected: boolean
  isModelLoaded: boolean
  isSearching: boolean
  isParked: boolean
  onReset: () => void
  onOpenMetrics: () => void
}

export const TopNav: React.FC<TopNavProps> = ({
  isConnected,
  isModelLoaded,
  isSearching,
  isParked,
  onReset,
  onOpenMetrics,
}) => {
  return (
    <header className="h-16 px-6 bg-[#0b101b]/95 backdrop-blur-md border-b border-[#1e293b] flex items-center justify-between z-30">
      {/* Brand & Subtitle */}
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold tracking-tight text-white font-sans">
              ParkNav <span className="text-cyan-400">AI</span>
            </h1>
            <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
              DQN Agent
            </span>
          </div>
          <p className="text-xs text-slate-400">Intelligent Parking Navigation Simulation</p>
        </div>
      </div>

      {/* Status Badges & Quick Actions */}
      <div className="flex items-center gap-3">
        {/* Backend Connectivity Status */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border ${
            isConnected
              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/80'
              : 'bg-rose-950/60 text-rose-300 border-rose-800/80'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
            }`}
          />
          {isConnected ? 'Backend Online' : 'Connecting...'}
        </div>

        {/* Model Checkpoint Status */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border ${
            isModelLoaded
              ? 'bg-blue-950/60 text-blue-300 border-blue-800/80'
              : 'bg-amber-950/60 text-amber-300 border-amber-800/80'
          }`}
        >
          <Cpu className="w-3.5 h-3.5" />
          {isModelLoaded ? 'PyTorch DDQN Loaded' : 'DQN model unavailable'}
        </div>

        {/* Simulation State Badge */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-900 text-slate-300 border border-slate-800">
          {isParked ? (
            <span className="flex items-center gap-1.5 text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> Parked Successfully
            </span>
          ) : isSearching ? (
            <span className="flex items-center gap-1.5 text-cyan-400 animate-pulse">
              <span className="w-2 h-2 rounded-full bg-cyan-400" /> Navigating via DQN...
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-slate-400">
              <AlertCircle className="w-3.5 h-3.5" /> Ready at Entry
            </span>
          )}
        </div>

        {/* Benchmark / Model Specs Dialog Trigger */}
        <button
          onClick={onOpenMetrics}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors flex items-center gap-1.5"
          title="View Model Specs & Evaluation Benchmarks"
        >
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>Benchmarks</span>
        </button>

        {/* Reset Simulation Button */}
        <button
          onClick={onReset}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors flex items-center gap-1.5"
          title="Reset Environment to Initial State"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset</span>
        </button>
      </div>
    </header>
  )
}
