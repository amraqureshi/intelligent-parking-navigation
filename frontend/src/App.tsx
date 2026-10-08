import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from './api/client'
import type {
  FacilityLayout,
  SimulationState,
  ActivityEvent,
  ModelMetrics,
} from './types/simulation'
import { TopNav } from './components/TopNav'
import { ParkingLotCanvas } from './components/ParkingLotCanvas'
import { gridToCanvas } from './utils/coordinates'
import { ControlsBar } from './components/ControlsBar'
import { MetricsPanel } from './components/MetricsPanel'
import { ActivityLog } from './components/ActivityLog'
import { AgentTelemetryModal } from './components/AgentTelemetry'
import { SuccessModal } from './components/SuccessModal'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'

export function App() {
  // Facility and environment states
  const [layout, setLayout] = useState<FacilityLayout | null>(null)
  const [state, setState] = useState<SimulationState | null>(null)
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [isConnected, setIsConnected] = useState<boolean>(false)
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false)

  // Simulation controls
  const [isSearching, setIsSearching] = useState<boolean>(false)
  const [isMoving, setIsMoving] = useState<boolean>(false)
  const [isPaused, setIsPaused] = useState<boolean>(false)
  const [simulationSpeed, setSimulationSpeed] = useState<number>(1)
  const [selectedScenario, setSelectedScenario] = useState<string>('medium')
  const [targetSpot, setTargetSpot] = useState<string | null>(null)
  const [plannedRoute, setPlannedRoute] = useState<string[]>(['ENTRY'])

  // Modals & Panels
  const [isMetricsModalOpen, setIsMetricsModalOpen] = useState<boolean>(false)
  const [isSuccessModalOpen, setIsSuccessModalOpen] = useState<boolean>(false)
  const [zoom, setZoom] = useState<number>(1.0)

  // Activity events
  const [events, setEvents] = useState<ActivityEvent[]>([])

  // Vehicle visual position and rotation
  const [vehiclePos, setVehiclePos] = useState<{ x: number; y: number; rotation: number }>(() => {
    const p = gridToCanvas(0, 1)
    return { x: p.x, y: p.y, rotation: 0 }
  })

  // Refs for animation loop
  const isPausedRef = useRef(isPaused)
  const isMovingRef = useRef(isMoving)
  const simulationSpeedRef = useRef(simulationSpeed)

  useEffect(() => {
    isPausedRef.current = isPaused
  }, [isPaused])

  useEffect(() => {
    isMovingRef.current = isMoving
  }, [isMoving])

  useEffect(() => {
    simulationSpeedRef.current = simulationSpeed
  }, [simulationSpeed])

  const addEvent = useCallback((type: ActivityEvent['type'], title: string, description: string) => {
    const newEvt: ActivityEvent = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toLocaleTimeString(),
      type,
      title,
      description,
    }
    setEvents((prev) => [newEvt, ...prev].slice(0, 50))
  }, [])

  // -------------------------------------------------------------------------
  // 1. Initial Load & Backend Health Check
  // -------------------------------------------------------------------------
  useEffect(() => {
    async function init() {
      try {
        const health = await api.getHealth()
        setIsConnected(true)
        setIsModelLoaded(health.model_loaded)

        const [layoutData, stateData, metricsData] = await Promise.all([
          api.getLayout(),
          api.getState(),
          api.getMetrics().catch(() => null),
        ])

        setLayout(layoutData)
        setState(stateData)
        setMetrics(metricsData)

        // Set vehicle initial coordinate
        const initPos = gridToCanvas(stateData.current_pos.x, stateData.current_pos.y)
        setVehiclePos({ x: initPos.x, y: initPos.y, rotation: 0 })

        addEvent(
          'info',
          'Facility Digital Twin Loaded',
          `Facility online with ${layoutData.parking_spots.length} bays and 29 graph nodes.`
        )
      } catch (err: unknown) {
        setIsConnected(false)
        const msg = err instanceof Error ? err.message : String(err)
        addEvent('warning', 'Connection Error', `Failed to connect to backend: ${msg}`)
      }
    }
    init()
  }, [addEvent])

  // -------------------------------------------------------------------------
  // 2. Smooth Vehicle Animation along Segments
  // -------------------------------------------------------------------------
  const animateSegment = useCallback(
    (
      fromCoord: { x: number; y: number },
      toCoord: { x: number; y: number },
      targetRotation: number,
      durationMs: number
    ): Promise<void> => {
      return new Promise((resolve) => {
        const startTime = performance.now()

        function tick(currentTime: number) {
          if (isPausedRef.current) {
            // While paused, wait and request next frame
            requestAnimationFrame(tick)
            return
          }

          const elapsed = (currentTime - startTime) * simulationSpeedRef.current
          const progress = Math.min(1.0, elapsed / durationMs)

          // Linear interpolation for smooth coordinate travel
          const curX = fromCoord.x + (toCoord.x - fromCoord.x) * progress
          const curY = fromCoord.y + (toCoord.y - fromCoord.y) * progress

          setVehiclePos({
            x: curX,
            y: curY,
            rotation: targetRotation,
          })

          if (progress < 1.0) {
            requestAnimationFrame(tick)
          } else {
            setVehiclePos({
              x: toCoord.x,
              y: toCoord.y,
              rotation: targetRotation,
            })
            resolve()
          }
        }

        requestAnimationFrame(tick)
      })
    },
    []
  )

  // -------------------------------------------------------------------------
  // 3. Sequential Route Navigation Runner
  // -------------------------------------------------------------------------
  const runRouteAnimation = useCallback(
    async (route: string[], destinationSpot: string | null) => {
      if (!layout || route.length < 2) return

      setIsMoving(true)
      setIsSearching(true)
      isMovingRef.current = true

      // Map node IDs to canvas coordinates
      const coordsMap = new Map<string, { x: number; y: number }>()
      for (const node of layout.nodes) {
        coordsMap.set(node.id, gridToCanvas(node.x, node.y))
      }

      for (let i = 0; i < route.length - 1; i++) {
        const fromId = route[i]
        const toId = route[i + 1]
        const fromP = coordsMap.get(fromId)
        const toP = coordsMap.get(toId)

        if (!fromP || !toP) continue

        // Calculate travel angle
        const dx = toP.x - fromP.x
        const dy = toP.y - fromP.y
        let angleDeg = 0
        if (dx > 0 && Math.abs(dy) < 5) angleDeg = 0 // East (RIGHT)
        else if (dy > 0 && Math.abs(dx) < 5) angleDeg = 90 // South (DOWN)
        else if (dx < 0 && Math.abs(dy) < 5) angleDeg = 180 // West (LEFT)
        else if (dy < 0 && Math.abs(dx) < 5) angleDeg = 270 // North (UP)
        else angleDeg = Math.atan2(dy, dx) * (180 / Math.PI)

        addEvent('action', `Traversing: ${fromId} → ${toId}`, `Heading ${angleDeg.toFixed(0)}° along aisle`)

        // Animate segment
        const baseDuration = 550
        await animateSegment(fromP, toP, angleDeg, baseDuration)

        // After completing segment, advance backend step to keep simulation synchronized
        try {
          const stepRes = await api.step()
          setState(stepRes.state)
        } catch {
          // Continue visual track
        }
      }

      setIsMoving(false)
      isMovingRef.current = false

      if (destinationSpot) {
        addEvent('success', 'Parking Complete!', `Vehicle parked successfully at stall ${destinationSpot}`)
        setIsSuccessModalOpen(true)
      }
    },
    [layout, animateSegment, addEvent]
  )

  // -------------------------------------------------------------------------
  // 4. Find Parking Trigger (Primary User Journey)
  // -------------------------------------------------------------------------
  const handleStartSearch = async () => {
    try {
      addEvent('dqn', 'Searching Parking...', 'Querying Deep Q-Network for optimal path.')
      const result = await api.startSearch()

      if (result.success && result.parking_spot) {
        setTargetSpot(result.parking_spot)
        setPlannedRoute(result.route)
        addEvent(
          'dqn',
          `Optimal Target: ${result.parking_spot}`,
          `DQN found path with ${result.steps} steps (${result.route.join(' → ')})`
        )
        // Run smooth animation along confirmed path
        await runRouteAnimation(result.route, result.parking_spot)
      } else {
        addEvent('warning', 'No Spot Found', result.message)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addEvent('warning', 'Search Failed', msg)
    }
  }

  // -------------------------------------------------------------------------
  // 5. Single Step Execution
  // -------------------------------------------------------------------------
  const handleStep = async () => {
    try {
      const res = await api.step()
      setState(res.state)

      if (layout) {
        const fromP = gridToCanvas(
          layout.nodes.find((n) => n.id === res.prev_node)?.x ?? 0,
          layout.nodes.find((n) => n.id === res.prev_node)?.y ?? 1
        )
        const toP = gridToCanvas(
          layout.nodes.find((n) => n.id === res.current_node)?.x ?? 0,
          layout.nodes.find((n) => n.id === res.current_node)?.y ?? 1
        )

        const dx = toP.x - fromP.x
        const dy = toP.y - fromP.y
        let angleDeg = 0
        if (dx > 0 && Math.abs(dy) < 5) angleDeg = 0
        else if (dy > 0 && Math.abs(dx) < 5) angleDeg = 90
        else if (dx < 0 && Math.abs(dy) < 5) angleDeg = 180
        else if (dy < 0 && Math.abs(dx) < 5) angleDeg = 270

        await animateSegment(fromP, toP, angleDeg, 350)
      }

      addEvent('action', `Step ${res.action_name}`, `Transitioned to node '${res.current_node}' (Reward: ${res.reward})`)

      if (res.success && res.parking_spot) {
        setIsSuccessModalOpen(true)
        addEvent('success', 'Parked at Destination', `Arrived at spot ${res.parking_spot}`)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addEvent('warning', 'Step Error', msg)
    }
  }

  // -------------------------------------------------------------------------
  // 6. Reset Simulation
  // -------------------------------------------------------------------------
  const handleReset = async () => {
    try {
      setIsPaused(false)
      setIsMoving(false)
      setIsSearching(false)
      setTargetSpot(null)
      setPlannedRoute(['ENTRY'])
      setIsSuccessModalOpen(false)

      const stateData = await api.reset({ start_node: 'ENTRY', random_occupancy_rate: 0.5 })
      setState(stateData)

      const entryPos = gridToCanvas(0, 1)
      setVehiclePos({ x: entryPos.x, y: entryPos.y, rotation: 0 })

      addEvent('info', 'Simulation Reset', 'Vehicle returned to Entry Gate (0, 1). Ready for navigation.')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addEvent('warning', 'Reset Error', msg)
    }
  }

  // -------------------------------------------------------------------------
  // 7. Interactive Spot Occupancy Toggle
  // -------------------------------------------------------------------------
  const handleToggleSpot = async (spotId: string) => {
    try {
      const updatedState = await api.toggleSpot(spotId)
      setState(updatedState)
      const nowStatus = updatedState.status_map[spotId]
      addEvent('info', `Bay ${spotId} Updated`, `Spot status toggled to ${nowStatus.toUpperCase()}`)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addEvent('warning', 'Toggle Error', msg)
    }
  }

  // -------------------------------------------------------------------------
  // 8. Scenario Preset Change
  // -------------------------------------------------------------------------
  const handleScenarioChange = async (preset: string) => {
    setSelectedScenario(preset)
    try {
      setIsPaused(false)
      setIsMoving(false)
      setIsSearching(false)
      setTargetSpot(null)
      setPlannedRoute(['ENTRY'])

      const updatedState = await api.setScenario(preset)
      setState(updatedState)

      const entryPos = gridToCanvas(0, 1)
      setVehiclePos({ x: entryPos.x, y: entryPos.y, rotation: 0 })

      addEvent('info', `Scenario: ${preset.toUpperCase()}`, `Configured with ${updatedState.occupied_count}/12 spots occupied.`)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addEvent('warning', 'Scenario Error', msg)
    }
  }

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* 1. Header Navigation */}
      <TopNav
        isConnected={isConnected}
        isModelLoaded={isModelLoaded}
        isSearching={isSearching}
        isParked={state?.is_parked ?? false}
        onReset={handleReset}
        onOpenMetrics={() => setIsMetricsModalOpen(true)}
      />

      {/* 2. Main Dashboard Area */}
      <main className="flex-1 flex flex-col lg:flex-row gap-5 p-5 max-w-[1720px] w-full mx-auto">
        {/* Left Column: Interactive Simulation Canvas */}
        <section className="flex-1 flex flex-col gap-4">
          {/* Canvas Container Card */}
          <div className="relative flex-1 bg-[#0b101b] border border-[#1e293b] rounded-2xl shadow-2xl overflow-hidden flex flex-col min-h-[580px]">
            {/* Canvas Status & Zoom Toolbar */}
            <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
              <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800 text-xs px-3 py-1.5 rounded-xl font-mono text-slate-300 flex items-center gap-2 shadow-lg">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                <span>GRID: 10 x 5 LANES</span>
              </div>
            </div>

            {/* Canvas Floating Zoom Buttons */}
            <div className="absolute top-4 right-4 z-20 flex items-center gap-1.5 bg-slate-900/90 backdrop-blur-md border border-slate-800 p-1 rounded-xl shadow-lg">
              <button
                onClick={() => setZoom((z) => Math.min(1.4, z + 0.1))}
                className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoom((z) => Math.max(0.7, z - 0.1))}
                className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoom(1.0)}
                className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                title="Fit to View"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>

            {/* Parking Lot Canvas Element */}
            <div className="flex-1 w-full h-full flex items-center justify-center p-2">
              <ParkingLotCanvas
                layout={layout}
                state={state}
                vehiclePos={vehiclePos}
                isMoving={isMoving}
                targetSpot={targetSpot}
                plannedRoute={plannedRoute}
                onToggleSpot={handleToggleSpot}
                zoom={zoom}
              />
            </div>

            {/* Bottom Canvas Legend */}
            <div className="bg-[#0f172a]/90 backdrop-blur-md border-t border-slate-800/80 px-6 py-2.5 flex flex-wrap items-center justify-between text-xs text-slate-400">
              <div className="flex items-center gap-5">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-emerald-500/20 border border-emerald-500" />
                  <span>Available</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-rose-500/20 border border-rose-500" />
                  <span>Occupied</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-cyan-500/30 border border-cyan-400 animate-pulse" />
                  <span>DQN Target</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-yellow-400" />
                  <span>Road Centerline</span>
                </div>
              </div>
              <div className="text-[11px] text-slate-500 font-mono">
                Click any bay to toggle state
              </div>
            </div>
          </div>

          {/* Primary Simulation Controls */}
          <ControlsBar
            isSearching={isSearching}
            isMoving={isMoving}
            isPaused={isPaused}
            isParked={state?.is_parked ?? false}
            simulationSpeed={simulationSpeed}
            selectedScenario={selectedScenario}
            availableCount={state?.available_count ?? 0}
            onStartSearch={handleStartSearch}
            onTogglePlayPause={() => setIsPaused((p) => !p)}
            onStep={handleStep}
            onReset={handleReset}
            onSpeedChange={setSimulationSpeed}
            onScenarioChange={handleScenarioChange}
          />
        </section>

        {/* Right Column: Telemetry & Activity Feed */}
        <aside className="w-full lg:w-96 flex flex-col gap-4">
          <MetricsPanel
            state={state}
            targetSpot={targetSpot}
            plannedRoute={plannedRoute}
          />

          <ActivityLog
            events={events}
            onClear={() => setEvents([])}
          />
        </aside>
      </main>

      {/* Model Specs & Telemetry Modal */}
      <AgentTelemetryModal
        state={state}
        metrics={metrics}
        isOpen={isMetricsModalOpen}
        onClose={() => setIsMetricsModalOpen(false)}
      />

      {/* Parking Goal Reached Modal */}
      <SuccessModal
        isOpen={isSuccessModalOpen}
        parkedSpot={state?.parked_spot ?? targetSpot}
        stepsTaken={state?.current_step ?? plannedRoute.length - 1}
        totalReward={state?.total_reward ?? 50.0}
        onClose={() => setIsSuccessModalOpen(false)}
        onReset={handleReset}
      />
    </div>
  )
}

export default App
