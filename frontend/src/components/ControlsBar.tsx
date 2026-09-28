import React from 'react'
import {
  Compass,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  SlidersHorizontal,
  Gauge,
} from 'lucide-react'

interface ControlsBarProps {
  isSearching: boolean
  isMoving: boolean
  isPaused: boolean
  isParked: boolean
  simulationSpeed: number
  selectedScenario: string
  availableCount: number
  onStartSearch: () => void
  onTogglePlayPause: () => void
  onStep: () => void
  onReset: () => void
  onSpeedChange: (speed: number) => void
  onScenarioChange: (preset: string) => void
}

export const ControlsBar: React.FC<ControlsBarProps> = ({
  isSearching,
  isMoving,
  isPaused,
  isParked,
  simulationSpeed,
  selectedScenario,
  availableCount,
  onStartSearch,
  onTogglePlayPause,
  onStep,
  onReset,
  onSpeedChange,
  onScenarioChange,
}) => {
  return (
    <div className="bg-[#0f172a]/95 backdrop-blur-md border border-[#1e293b] rounded-2xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
      {/* Primary Action Button: FIND PARKING */}
      <div className="flex items-center gap-3">
        <button
          onClick={onStartSearch}
          disabled={isMoving && !isPaused}
          className={`relative group px-6 py-3 rounded-xl font-bold text-sm tracking-wide transition-all duration-300 flex items-center gap-2.5 shadow-lg ${
            isMoving && !isPaused
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
              : 'bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:scale-[1.02] active:scale-[0.98]'
          }`}
        >
          <Compass className={`w-5 h-5 ${isMoving ? 'animate-spin' : 'group-hover:rotate-45 transition-transform'}`} />
          <span>{isSearching && isMoving ? 'Navigating via DQN...' : 'Find Parking Spot'}</span>
          {availableCount > 0 && !isMoving && (
            <span className="ml-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-white/20 text-white">
              {availableCount} free
            </span>
          )}
        </button>

        {/* Secondary Step / Playback Controls */}
        <div className="flex items-center gap-1.5 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800">
          {/* Play / Pause Toggle */}
          <button
            onClick={onTogglePlayPause}
            disabled={!isSearching || isParked}
            className={`p-2.5 rounded-lg text-xs font-medium transition-colors ${
              !isSearching || isParked
                ? 'text-slate-600 cursor-not-allowed'
                : isPaused
                ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                : 'bg-amber-600/30 hover:bg-amber-600/40 text-amber-300 border border-amber-600/40'
            }`}
            title={isPaused ? 'Resume Navigation' : 'Pause Navigation'}
          >
            {isPaused ? <Play className="w-4 h-4 fill-current" /> : <Pause className="w-4 h-4 fill-current" />}
          </button>

          {/* Single Step (Manual Discrete Step) */}
          <button
            onClick={onStep}
            disabled={isMoving && !isPaused}
            className="p-2.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors disabled:text-slate-600 disabled:cursor-not-allowed"
            title="Execute Single DQN Action Step"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Reset */}
          <button
            onClick={onReset}
            className="p-2.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            title="Reset Simulation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Speed & Scenario Controls */}
      <div className="flex items-center flex-wrap gap-4">
        {/* Speed Adjustment */}
        <div className="flex items-center gap-2 bg-slate-900/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
          <Gauge className="w-4 h-4 text-slate-400" />
          <span className="text-slate-400 font-medium mr-1">Speed:</span>
          {[0.5, 1, 2, 4].map((spd) => (
            <button
              key={spd}
              onClick={() => onSpeedChange(spd)}
              className={`px-2 py-1 rounded-md font-semibold transition-colors ${
                simulationSpeed === spd
                  ? 'bg-cyan-500 text-slate-950 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {spd}x
            </button>
          ))}
        </div>

        {/* Scenario Presets Dropdown */}
        <div className="flex items-center gap-2 bg-slate-900/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
          <SlidersHorizontal className="w-4 h-4 text-slate-400" />
          <span className="text-slate-400 font-medium">Scenario:</span>
          <select
            value={selectedScenario}
            onChange={(e) => onScenarioChange(e.target.value)}
            disabled={isMoving && !isPaused}
            className="bg-slate-800 text-slate-200 font-medium rounded-lg px-2.5 py-1 border border-slate-700 outline-none focus:border-cyan-500 cursor-pointer disabled:opacity-50"
          >
            <option value="medium">50% Occupied (Medium)</option>
            <option value="low">25% Occupied (Light)</option>
            <option value="high">75% Occupied (Heavy)</option>
            <option value="aisle1_full">Aisle 1 Full (P01-P06)</option>
            <option value="aisle2_full">Aisle 2 Full (P07-P12)</option>
            <option value="random">Random Sampling</option>
            <option value="empty">100% Free Lot</option>
          </select>
        </div>
      </div>
    </div>
  )
}
