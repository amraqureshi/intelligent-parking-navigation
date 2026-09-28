import React from 'react'
import { Brain, Layers, ShieldCheck, X } from 'lucide-react'
import type { SimulationState, ModelMetrics } from '../types/simulation'

interface AgentTelemetryProps {
  state: SimulationState | null
  metrics: ModelMetrics | null
  isOpen: boolean
  onClose: () => void
}

export const AgentTelemetryModal: React.FC<AgentTelemetryProps> = ({
  state,
  metrics: _metrics,
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null

  const validActionSet = new Set(state?.valid_actions.map((a) => a.action) ?? [])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#0f172a] border border-[#1e293b] rounded-2xl w-full max-w-3xl max-h-[85vh] overflow-y-auto shadow-2xl p-6 flex flex-col gap-6 custom-scrollbar">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-950 border border-cyan-800 flex items-center justify-center text-cyan-400">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                Deep Q-Network (DDQN) Intelligence Engine
              </h2>
              <p className="text-xs text-slate-400">
                Neural Policy Telemetry & Benchmark Evaluation
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 1. Real-time Action Space & Action Masking */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" /> Action Space & Masking at Current Node ({state?.current_node || 'ENTRY'})
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { id: 0, name: 'UP', dy: '+1 (North)', key: 'ArrowUp' },
              { id: 1, name: 'DOWN', dy: '-1 (South)', key: 'ArrowDown' },
              { id: 2, name: 'LEFT', dy: '-1 (West)', key: 'ArrowLeft' },
              { id: 3, name: 'RIGHT', dy: '+1 (East)', key: 'ArrowRight' },
            ].map((act) => {
              const isValid = validActionSet.has(act.id)
              const isLast = state?.last_action === act.name

              return (
                <div
                  key={act.id}
                  className={`p-3 rounded-xl border transition-all ${
                    isLast
                      ? 'bg-cyan-950/60 border-cyan-500 shadow-md shadow-cyan-500/20'
                      : isValid
                      ? 'bg-slate-900/80 border-slate-700/80'
                      : 'bg-slate-900/30 border-slate-800/40 opacity-40'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-bold text-white font-mono">
                      Action {act.id}
                    </span>
                    <span
                      className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        isValid
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                          : 'bg-rose-950 text-rose-400 border border-rose-800'
                      }`}
                    >
                      {isValid ? 'VALID' : 'MASKED'}
                    </span>
                  </div>
                  <div className="text-sm font-black text-cyan-400 font-mono">
                    {act.name}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">{act.dy}</div>
                </div>
              )
            })}
          </div>
        </div>

        {/* 2. Neural Network Architecture Breakdown */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" /> PyTorch Network Architecture (47 → 128 → 128 → 64 → 4)
          </h3>
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3 text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-slate-800/60 p-2.5 rounded-lg">
                <span className="text-slate-400 text-[10px]">State Vector Dimension:</span>
                <p className="font-mono font-bold text-cyan-300 text-sm mt-0.5">47 Floats</p>
                <span className="text-[10px] text-slate-500">29 Node + 12 Spot + Coords + Dist</span>
              </div>
              <div className="bg-slate-800/60 p-2.5 rounded-lg">
                <span className="text-slate-400 text-[10px]">Hidden Layers:</span>
                <p className="font-mono font-bold text-cyan-300 text-sm mt-0.5">3 Layers (ReLU)</p>
                <span className="text-[10px] text-slate-500">128 → 128 → 64 Units</span>
              </div>
              <div className="bg-slate-800/60 p-2.5 rounded-lg">
                <span className="text-slate-400 text-[10px]">Policy Algorithm:</span>
                <p className="font-mono font-bold text-cyan-300 text-sm mt-0.5">Double DQN</p>
                <span className="text-[10px] text-slate-500">Decoupled Action/Value</span>
              </div>
              <div className="bg-slate-800/60 p-2.5 rounded-lg">
                <span className="text-slate-400 text-[10px]">Target Synchronization:</span>
                <p className="font-mono font-bold text-cyan-300 text-sm mt-0.5">Polyak Soft Update</p>
                <span className="text-[10px] text-slate-500">τ = 0.005</span>
              </div>
            </div>
          </div>
        </div>

        {/* 3. 400-Episode Benchmark Comparison Results */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-400" /> Ground-Truth Benchmark Results (Dijkstra vs. Trained DDQN)
          </h3>
          <div className="border border-slate-800 rounded-xl overflow-hidden text-xs">
            <table className="w-full text-left">
              <thead className="bg-slate-900/90 text-slate-400 text-[10px] uppercase font-mono border-b border-slate-800">
                <tr>
                  <th className="p-3">Congestion Level</th>
                  <th className="p-3">Policy Method</th>
                  <th className="p-3">Success Rate</th>
                  <th className="p-3">Avg Steps</th>
                  <th className="p-3">Avg Reward</th>
                  <th className="p-3">Invalid Moves</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 font-mono">
                {/* 25% Row */}
                <tr className="bg-slate-900/30">
                  <td className="p-3 font-bold text-slate-200" rowSpan={2}>
                    25% (Light)
                  </td>
                  <td className="p-3 text-slate-400">Baseline (Dijkstra)</td>
                  <td className="p-3 text-emerald-400 font-bold">100.0%</td>
                  <td className="p-3 text-slate-200">3.02</td>
                  <td className="p-3 text-emerald-400">+50.91</td>
                  <td className="p-3 text-slate-400">0</td>
                </tr>
                <tr className="bg-cyan-950/20 text-cyan-300">
                  <td className="p-3 font-bold text-cyan-300">DQN Agent (PyTorch)</td>
                  <td className="p-3 text-emerald-400 font-bold">100.0%</td>
                  <td className="p-3 font-bold text-cyan-300">3.02</td>
                  <td className="p-3 font-bold text-emerald-400">+50.91</td>
                  <td className="p-3 text-emerald-400 font-bold">0</td>
                </tr>

                {/* 50% Row */}
                <tr className="bg-slate-900/30">
                  <td className="p-3 font-bold text-slate-200" rowSpan={2}>
                    50% (Medium)
                  </td>
                  <td className="p-3 text-slate-400">Baseline (Dijkstra)</td>
                  <td className="p-3 text-emerald-400 font-bold">100.0%</td>
                  <td className="p-3 text-slate-200">3.18</td>
                  <td className="p-3 text-emerald-400">+50.95</td>
                  <td className="p-3 text-slate-400">0</td>
                </tr>
                <tr className="bg-cyan-950/20 text-cyan-300">
                  <td className="p-3 font-bold text-cyan-300">DQN Agent (PyTorch)</td>
                  <td className="p-3 text-emerald-400 font-bold">100.0%</td>
                  <td className="p-3 font-bold text-cyan-300">3.18</td>
                  <td className="p-3 font-bold text-emerald-400">+50.95</td>
                  <td className="p-3 text-emerald-400 font-bold">0</td>
                </tr>

                {/* 75% Row */}
                <tr className="bg-slate-900/30">
                  <td className="p-3 font-bold text-slate-200" rowSpan={2}>
                    75% (Heavy)
                  </td>
                  <td className="p-3 text-slate-400">Baseline (Dijkstra)</td>
                  <td className="p-3 text-emerald-400 font-bold">100.0%</td>
                  <td className="p-3 text-slate-200">3.80</td>
                  <td className="p-3 text-emerald-400">+51.14</td>
                  <td className="p-3 text-slate-400">0</td>
                </tr>
                <tr className="bg-cyan-950/20 text-cyan-300">
                  <td className="p-3 font-bold text-cyan-300">DQN Agent (PyTorch)</td>
                  <td className="p-3 text-emerald-400 font-bold">92.0%</td>
                  <td className="p-3 font-bold text-cyan-300">7.54</td>
                  <td className="p-3 font-bold text-emerald-400">+45.42</td>
                  <td className="p-3 text-emerald-400 font-bold">0</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-3 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white transition-colors"
          >
            Close Telemetry
          </button>
        </div>
      </div>
    </div>
  )
}
