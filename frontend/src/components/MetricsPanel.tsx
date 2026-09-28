import React from 'react'
import {
  Car,
  CheckCircle,
  XCircle,
  MapPin,
  TrendingUp,
  Award,
  Navigation,
  Footprints,
} from 'lucide-react'
import type { SimulationState } from '../types/simulation'

interface MetricsPanelProps {
  state: SimulationState | null
  targetSpot: string | null
  plannedRoute: string[]
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  state,
  targetSpot,
  plannedRoute,
}) => {
  const totalSpots = state?.total_spots ?? 12
  const availableCount = state?.available_count ?? 6
  const occupiedCount = state?.occupied_count ?? 6
  const occupancyRate = state?.occupancy_rate ?? 0.5
  const occupancyPct = Math.round(occupancyRate * 100)

  const stepsTaken = state?.current_step ?? 0
  const maxSteps = state?.max_steps ?? 50
  const totalReward = state?.total_reward ?? 0.0

  return (
    <div className="bg-[#0f172a]/95 backdrop-blur-md border border-[#1e293b] rounded-2xl p-5 shadow-xl flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Car className="w-5 h-5 text-cyan-400" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">
            Facility Telemetry
          </h2>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-md">
          {totalSpots} SPOTS TOTAL
        </span>
      </div>

      {/* Primary KPI Grid: Available vs Occupied */}
      <div className="grid grid-cols-2 gap-3">
        {/* Available Card */}
        <div className="bg-emerald-950/20 border border-emerald-900/40 rounded-xl p-3 flex items-center justify-between">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-wider text-emerald-400/80 flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> Available
            </div>
            <div className="text-2xl font-black text-emerald-300 font-mono mt-0.5">
              {availableCount}
            </div>
          </div>
          <div className="w-9 h-9 rounded-lg bg-emerald-900/30 flex items-center justify-center text-emerald-400 font-bold text-xs">
            {Math.round((availableCount / totalSpots) * 100)}%
          </div>
        </div>

        {/* Occupied Card */}
        <div className="bg-rose-950/20 border border-rose-900/40 rounded-xl p-3 flex items-center justify-between">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-wider text-rose-400/80 flex items-center gap-1.5">
              <XCircle className="w-3.5 h-3.5 text-rose-400" /> Occupied
            </div>
            <div className="text-2xl font-black text-rose-300 font-mono mt-0.5">
              {occupiedCount}
            </div>
          </div>
          <div className="w-9 h-9 rounded-lg bg-rose-900/30 flex items-center justify-center text-rose-400 font-bold text-xs">
            {occupancyPct}%
          </div>
        </div>
      </div>

      {/* Occupancy Progress Bar */}
      <div>
        <div className="flex justify-between items-center text-xs mb-1.5 font-medium">
          <span className="text-slate-400 flex items-center gap-1">
            <TrendingUp className="w-3.5 h-3.5 text-cyan-400" /> Congestion Level
          </span>
          <span
            className={`font-mono font-bold ${
              occupancyPct >= 75
                ? 'text-rose-400'
                : occupancyPct >= 50
                ? 'text-amber-400'
                : 'text-emerald-400'
            }`}
          >
            {occupancyPct}%
          </span>
        </div>
        <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              occupancyPct >= 75
                ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                : occupancyPct >= 50
                ? 'bg-gradient-to-r from-cyan-500 to-amber-500'
                : 'bg-gradient-to-r from-emerald-500 to-teal-400'
            }`}
            style={{ width: `${occupancyPct}%` }}
          />
        </div>
      </div>

      {/* Navigation Telemetry Cards */}
      <div className="space-y-2.5 pt-1 border-t border-slate-800/80 text-xs">
        {/* Vehicle Current Position */}
        <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-2">
            <MapPin className="w-4 h-4 text-cyan-400" /> Current Node:
          </span>
          <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded">
            {state?.current_node || 'ENTRY'}{' '}
            <span className="text-slate-400 text-[10px]">
              ({state?.current_pos.x ?? 0}, {state?.current_pos.y ?? 1})
            </span>
          </span>
        </div>

        {/* Target Parking Space */}
        <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-2">
            <Navigation className="w-4 h-4 text-blue-400" /> Target Destination:
          </span>
          <span
            className={`font-mono font-bold px-2 py-0.5 rounded ${
              targetSpot
                ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                : 'text-slate-500 bg-slate-800/50'
            }`}
          >
            {targetSpot || 'None (Searching...)'}
          </span>
        </div>

        {/* Steps Elapsed / Budget */}
        <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-2">
            <Footprints className="w-4 h-4 text-amber-400" /> Navigation Steps:
          </span>
          <span className="font-mono font-bold text-slate-200">
            {stepsTaken} <span className="text-slate-500 font-normal">/ {maxSteps}</span>
            {plannedRoute.length > 0 && !state?.is_parked && (
              <span className="ml-1 text-cyan-400 font-normal text-[11px]">
                ({Math.max(0, plannedRoute.length - 1 - stepsTaken)} left)
              </span>
            )}
          </span>
        </div>

        {/* Cumulative Reward */}
        <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-2">
            <Award className="w-4 h-4 text-yellow-400" /> Cumulative Reward:
          </span>
          <span
            className={`font-mono font-bold px-2 py-0.5 rounded ${
              totalReward > 0
                ? 'text-emerald-400 bg-emerald-950/40'
                : totalReward < 0
                ? 'text-rose-400 bg-rose-950/40'
                : 'text-slate-300 bg-slate-800'
            }`}
          >
            {totalReward > 0 ? `+${totalReward.toFixed(1)}` : totalReward.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Interactive Helper Hint */}
      <div className="text-[11px] text-slate-400 bg-slate-900/40 border border-slate-800/60 rounded-xl p-2.5 text-center">
        💡 <strong className="text-slate-300">Interactive:</strong> Click any bay (P01–P12) on the map to toggle its occupancy status in real time.
      </div>
    </div>
  )
}
