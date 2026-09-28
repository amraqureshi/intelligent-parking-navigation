import React from 'react'
import { Activity, CheckCircle, AlertTriangle, ArrowRight, BrainCircuit, Trash2 } from 'lucide-react'
import type { ActivityEvent } from '../types/simulation'

interface ActivityLogProps {
  events: ActivityEvent[]
  onClear: () => void
}

export const ActivityLog: React.FC<ActivityLogProps> = ({ events, onClear }) => {
  const getEventIcon = (type: ActivityEvent['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
      case 'warning':
        return <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
      case 'dqn':
        return <BrainCircuit className="w-3.5 h-3.5 text-cyan-400" />
      case 'action':
        return <ArrowRight className="w-3.5 h-3.5 text-blue-400" />
      default:
        return <Activity className="w-3.5 h-3.5 text-slate-400" />
    }
  }

  const getEventBadge = (type: ActivityEvent['type']) => {
    switch (type) {
      case 'success':
        return 'bg-emerald-950/60 text-emerald-300 border-emerald-800/80'
      case 'warning':
        return 'bg-amber-950/60 text-amber-300 border-amber-800/80'
      case 'dqn':
        return 'bg-cyan-950/60 text-cyan-300 border-cyan-800/80'
      case 'action':
        return 'bg-blue-950/60 text-blue-300 border-blue-800/80'
      default:
        return 'bg-slate-800/80 text-slate-300 border-slate-700/80'
    }
  }

  return (
    <div className="bg-[#0f172a]/95 backdrop-blur-md border border-[#1e293b] rounded-2xl p-5 shadow-xl flex flex-col h-[280px]">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Live Activity Feed
          </h3>
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
            {events.length}
          </span>
        </div>
        <button
          onClick={onClear}
          className="text-slate-500 hover:text-slate-300 text-xs p-1 rounded hover:bg-slate-800 transition-colors"
          title="Clear Event Log"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Events Scroll Area */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1.5 custom-scrollbar">
        {events.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-500 italic">
            No activity yet. Click &quot;Find Parking Spot&quot; to begin.
          </div>
        ) : (
          events.map((evt) => (
            <div
              key={evt.id}
              className="flex items-start gap-2.5 p-2 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-colors text-xs"
            >
              <div className="mt-0.5">{getEventIcon(evt.type)}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-200 truncate">{evt.title}</span>
                  <span className="text-[10px] text-slate-500 font-mono shrink-0">
                    {evt.timestamp}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-0.5 leading-snug">{evt.description}</p>
              </div>
              <span
                className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border font-mono shrink-0 ${getEventBadge(
                  evt.type
                )}`}
              >
                {evt.type}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
