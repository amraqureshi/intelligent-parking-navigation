import React from 'react'
import { CheckCircle2, Award, Navigation, Footprints, RotateCcw } from 'lucide-react'

interface SuccessModalProps {
  isOpen: boolean
  parkedSpot: string | null
  stepsTaken: number
  totalReward: number
  onClose: () => void
  onReset: () => void
}

export const SuccessModal: React.FC<SuccessModalProps> = ({
  isOpen,
  parkedSpot,
  stepsTaken,
  totalReward,
  onClose,
  onReset,
}) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#0f172a] border border-emerald-500/50 rounded-2xl w-full max-w-md shadow-2xl p-6 flex flex-col items-center text-center gap-4 relative overflow-hidden">
        {/* Ambient Top Glow */}
        <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-48 h-48 bg-emerald-500/20 rounded-full blur-3xl pointer-events-none" />

        {/* Success Icon */}
        <div className="w-16 h-16 rounded-2xl bg-emerald-950/80 border border-emerald-500/50 flex items-center justify-center text-emerald-400 shadow-lg shadow-emerald-500/30">
          <CheckCircle2 className="w-8 h-8" />
        </div>

        {/* Title */}
        <div>
          <span className="text-[11px] font-bold uppercase tracking-widest text-emerald-400 bg-emerald-950/60 px-2.5 py-0.5 rounded-full border border-emerald-800">
            DQN Goal Reached
          </span>
          <h2 className="text-xl font-bold text-white mt-1.5 font-sans">
            Successfully Parked at {parkedSpot || 'Destination'}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Autonomous vehicle completed optimal path navigation using the trained Deep Q-Network.
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 w-full my-1">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 text-left">
            <span className="text-[11px] text-slate-400 flex items-center gap-1.5">
              <Footprints className="w-3.5 h-3.5 text-cyan-400" /> Total Steps
            </span>
            <div className="text-xl font-black text-cyan-300 font-mono mt-0.5">
              {stepsTaken}
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 text-left">
            <span className="text-[11px] text-slate-400 flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5 text-yellow-400" /> Cumulative Reward
            </span>
            <div className="text-xl font-black text-emerald-300 font-mono mt-0.5">
              +{totalReward.toFixed(1)}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3 w-full mt-2">
          <button
            onClick={onReset}
            className="flex-1 py-2.5 px-4 rounded-xl text-xs font-bold text-slate-300 bg-slate-800 hover:bg-slate-700 hover:text-white transition-colors flex items-center justify-center gap-2 border border-slate-700"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Reset Simulation
          </button>
          <button
            onClick={onClose}
            className="flex-1 py-2.5 px-4 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 transition-all shadow-lg shadow-emerald-500/25 flex items-center justify-center gap-2"
          >
            <Navigation className="w-3.5 h-3.5" /> Continue
          </button>
        </div>
      </div>
    </div>
  )
}
