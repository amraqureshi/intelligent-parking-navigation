import React, { useMemo } from 'react'
import type { FacilityLayout, SimulationState } from '../types/simulation'
import { VehicleSprite } from './VehicleSprite'

interface ParkingLotCanvasProps {
  layout: FacilityLayout | null
  state: SimulationState | null
  vehiclePos: { x: number; y: number; rotation: number }
  isMoving: boolean
  targetSpot: string | null
  plannedRoute: string[]
  onToggleSpot: (spotId: string) => void
  zoom: number
}

// Coordinate mapping constants
const CANVAS_WIDTH = 1060
const CANVAS_HEIGHT = 620
const PADDING_X = 95
const PADDING_Y = 85
const X_STEP = 87
const Y_STEP = 112.5

export function gridToCanvas(gx: number, gy: number): { x: number; y: number } {
  const x = PADDING_X + gx * X_STEP
  // Invert Y so gy=5 is top, gy=1 is bottom
  const y = PADDING_Y + (5 - gy) * Y_STEP
  return { x, y }
}

export const ParkingLotCanvas: React.FC<ParkingLotCanvasProps> = ({
  layout,
  state,
  vehiclePos,
  isMoving,
  targetSpot,
  plannedRoute,
  onToggleSpot,
  zoom,
}) => {
  // Pre-calculate node pixel positions map
  const nodeCoords = useMemo(() => {
    const map = new Map<string, { x: number; y: number; type: string }>()
    if (!layout) return map
    for (const node of layout.nodes) {
      const coords = gridToCanvas(node.x, node.y)
      map.set(node.id, { ...coords, type: node.type })
    }
    return map
  }, [layout])

  // Build planned route SVG path points
  const plannedPathD = useMemo(() => {
    if (!plannedRoute || plannedRoute.length < 2) return ''
    const points: string[] = []
    for (const nodeId of plannedRoute) {
      const coord = nodeCoords.get(nodeId)
      if (coord) {
        points.push(`${coord.x},${coord.y}`)
      }
    }
    if (points.length < 2) return ''
    return 'M ' + points.join(' L ')
  }, [plannedRoute, nodeCoords])

  // Pre-calculate parking spot data
  const spotsData = useMemo(() => {
    const spots: Array<{
      id: string
      x: number
      y: number
      gx: number
      gy: number
      isOccupied: boolean
      isTarget: boolean
      aisleSide: 'left' | 'right'
    }> = []

    if (!layout || !state) return spots

    for (const spotId of layout.parking_spots) {
      const node = layout.nodes.find((n) => n.id === spotId)
      if (!node) continue
      const coords = gridToCanvas(node.x, node.y)
      const isOccupied = state.status_map[spotId] === 'occupied'
      const isTarget = targetSpot === spotId

      // Determine stall orientation based on x coordinate
      // x=1 or x=4 are on left of aisle (pointing west)
      // x=3 or x=6 are on right of aisle (pointing east)
      const aisleSide = node.x === 1 || node.x === 4 ? 'left' : 'right'

      spots.push({
        id: spotId,
        x: coords.x,
        y: coords.y,
        gx: node.x,
        gy: node.y,
        isOccupied,
        isTarget,
        aisleSide,
      })
    }
    return spots
  }, [layout, state, targetSpot])

  return (
    <div className="relative w-full h-full flex items-center justify-center overflow-hidden select-none bg-[#090d16]">
      <svg
        viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
        className="w-full h-full max-h-[720px] transition-transform duration-200 ease-out"
        style={{ transform: `scale(${zoom})` }}
      >
        <defs>
          {/* Asphalt noise/gradient */}
          <radialGradient id="asphaltGradient" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor="#172033" />
            <stop offset="100%" stopColor="#0e1524" />
          </radialGradient>

          {/* Road asphalt surface fill */}
          <linearGradient id="roadSurface" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#1e293b" />
            <stop offset="50%" stopColor="#182234" />
            <stop offset="100%" stopColor="#141c2c" />
          </linearGradient>

          {/* Headlight cone beam gradient */}
          <linearGradient id="headlightGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.8" />
            <stop offset="40%" stopColor="#38bdf8" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
          </linearGradient>

          {/* Vehicle chassis metallic gradient */}
          <linearGradient id="carBodyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="40%" stopColor="#0284c7" />
            <stop offset="100%" stopColor="#0369a1" />
          </linearGradient>

          {/* Sunroof tint gradient */}
          <linearGradient id="glassGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#1e293b" />
            <stop offset="50%" stopColor="#0ea5e9" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#0284c7" stopOpacity="0.6" />
          </linearGradient>

          {/* Neon path glow filter */}
          <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Subtle drop shadow */}
          <filter id="subtleShadow" x="-10%" y="-10%" width="120%" height="120%">
            <feDropShadow dx="0" dy="4" stdDeviation="4" floodColor="#000000" floodOpacity="0.6" />
          </filter>
        </defs>

        {/* ------------------------------------------------------------- */}
        {/* 1. Facility Foundation & Asphalt Yard */}
        {/* ------------------------------------------------------------- */}
        <rect width={CANVAS_WIDTH} height={CANVAS_HEIGHT} fill="url(#asphaltGradient)" />

        {/* Facility Border Curb & Greenery Barrier */}
        <rect
          x="30"
          y="25"
          width={CANVAS_WIDTH - 60}
          height={CANVAS_HEIGHT - 50}
          rx="18"
          fill="none"
          stroke="#1e293b"
          strokeWidth="6"
        />
        <rect
          x="34"
          y="29"
          width={CANVAS_WIDTH - 68}
          height={CANVAS_HEIGHT - 58}
          rx="14"
          fill="none"
          stroke="#064e3b"
          strokeWidth="3"
          opacity="0.4"
        />

        {/* Grid Coordinates watermark */}
        <g opacity="0.15" fontSize="10" fill="#94a3b8" fontFamily="monospace">
          <text x="50" y="55">TOP THOROUGHFARE (Y=5)</text>
          <text x="50" y={CANVAS_HEIGHT - 40}>BOTTOM THOROUGHFARE (Y=1)</text>
          <text x={gridToCanvas(2, 0).x - 25} y="45">AISLE 1</text>
          <text x={gridToCanvas(5, 0).x - 25} y="45">AISLE 2</text>
          <text x={gridToCanvas(8, 0).x - 45} y="45">AISLE 3 (BYPASS)</text>
        </g>

        {/* ------------------------------------------------------------- */}
        {/* 2. Realistic Driving Lanes (Asphalt Corridors) */}
        {/* ------------------------------------------------------------- */}
        {/* Lower Main Road (y=1: ENTRY -> J1 -> J2 -> J3 -> EXIT) */}
        {(() => {
          const p1 = gridToCanvas(0, 1)
          const p2 = gridToCanvas(10, 1)
          return (
            <rect
              x={p1.x - 30}
              y={p1.y - 28}
              width={p2.x - p1.x + 60}
              height={56}
              rx="8"
              fill="url(#roadSurface)"
              stroke="#334155"
              strokeWidth="1.5"
            />
          )
        })()}

        {/* Upper Main Road (y=5: J4 <-> J5 <-> J6) */}
        {(() => {
          const p1 = gridToCanvas(2, 5)
          const p2 = gridToCanvas(8, 5)
          return (
            <rect
              x={p1.x - 28}
              y={p1.y - 28}
              width={p2.x - p1.x + 56}
              height={56}
              rx="8"
              fill="url(#roadSurface)"
              stroke="#334155"
              strokeWidth="1.5"
            />
          )
        })()}

        {/* Aisle 1 Road (x=2: J1 <-> N1 <-> N2 <-> N3 <-> J4) */}
        {(() => {
          const top = gridToCanvas(2, 5)
          const bot = gridToCanvas(2, 1)
          return (
            <rect
              x={top.x - 28}
              y={top.y - 28}
              width={56}
              height={bot.y - top.y + 56}
              rx="8"
              fill="url(#roadSurface)"
              stroke="#334155"
              strokeWidth="1.5"
            />
          )
        })()}

        {/* Aisle 2 Road (x=5: J2 <-> N4 <-> N5 <-> N6 <-> J5) */}
        {(() => {
          const top = gridToCanvas(5, 5)
          const bot = gridToCanvas(5, 1)
          return (
            <rect
              x={top.x - 28}
              y={top.y - 28}
              width={56}
              height={bot.y - top.y + 56}
              rx="8"
              fill="url(#roadSurface)"
              stroke="#334155"
              strokeWidth="1.5"
            />
          )
        })()}

        {/* Aisle 3 Bypass Road (x=8: J3 <-> N7 <-> N8 <-> N9 <-> J6) */}
        {(() => {
          const top = gridToCanvas(8, 5)
          const bot = gridToCanvas(8, 1)
          return (
            <rect
              x={top.x - 28}
              y={top.y - 28}
              width={56}
              height={bot.y - top.y + 56}
              rx="8"
              fill="url(#roadSurface)"
              stroke="#334155"
              strokeWidth="1.5"
            />
          )
        })()}

        {/* ------------------------------------------------------------- */}
        {/* 3. Road Markings & Centerlines */}
        {/* ------------------------------------------------------------- */}
        {/* Lower Road Dashed Yellow Centerline */}
        {(() => {
          const p1 = gridToCanvas(0, 1)
          const p2 = gridToCanvas(10, 1)
          return (
            <line
              x1={p1.x - 15}
              y1={p1.y}
              x2={p2.x + 15}
              y2={p2.y}
              stroke="#eab308"
              strokeWidth="2"
              strokeDasharray="9 7"
              opacity="0.8"
            />
          )
        })()}

        {/* Upper Road Dashed Yellow Centerline */}
        {(() => {
          const p1 = gridToCanvas(2, 5)
          const p2 = gridToCanvas(8, 5)
          return (
            <line
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke="#eab308"
              strokeWidth="2"
              strokeDasharray="9 7"
              opacity="0.8"
            />
          )
        })()}

        {/* Aisle 1 Dashed Centerline */}
        {(() => {
          const top = gridToCanvas(2, 5)
          const bot = gridToCanvas(2, 1)
          return (
            <line
              x1={top.x}
              y1={top.y}
              x2={bot.x}
              y2={bot.y}
              stroke="#eab308"
              strokeWidth="2"
              strokeDasharray="9 7"
              opacity="0.8"
            />
          )
        })()}

        {/* Aisle 2 Dashed Centerline */}
        {(() => {
          const top = gridToCanvas(5, 5)
          const bot = gridToCanvas(5, 1)
          return (
            <line
              x1={top.x}
              y1={top.y}
              x2={bot.x}
              y2={bot.y}
              stroke="#eab308"
              strokeWidth="2"
              strokeDasharray="9 7"
              opacity="0.8"
            />
          )
        })()}

        {/* Aisle 3 Dashed Centerline */}
        {(() => {
          const top = gridToCanvas(8, 5)
          const bot = gridToCanvas(8, 1)
          return (
            <line
              x1={top.x}
              y1={top.y}
              x2={bot.x}
              y2={bot.y}
              stroke="#eab308"
              strokeWidth="2"
              strokeDasharray="9 7"
              opacity="0.8"
            />
          )
        })()}

        {/* Asphalt Directional Arrows (Road Markings) */}
        {/* Entry -> J1 arrow */}
        <path
          d={`M ${gridToCanvas(1, 1).x - 10} ${gridToCanvas(1, 1).y - 8} L ${gridToCanvas(1, 1).x + 10} ${gridToCanvas(1, 1).y} L ${gridToCanvas(1, 1).x - 10} ${gridToCanvas(1, 1).y + 8}`}
          stroke="#f8fafc"
          strokeWidth="2"
          fill="none"
          opacity="0.35"
        />
        {/* J3 -> Exit arrow */}
        <path
          d={`M ${gridToCanvas(9, 1).x - 10} ${gridToCanvas(9, 1).y - 8} L ${gridToCanvas(9, 1).x + 10} ${gridToCanvas(9, 1).y} L ${gridToCanvas(9, 1).x - 10} ${gridToCanvas(9, 1).y + 8}`}
          stroke="#f8fafc"
          strokeWidth="2"
          fill="none"
          opacity="0.35"
        />

        {/* ------------------------------------------------------------- */}
        {/* 4. Parking Bays (Stalls P01 - P12) */}
        {/* ------------------------------------------------------------- */}
        {spotsData.map((spot) => {
          const STALL_W = 62
          const STALL_H = 46
          const isTarget = spot.isTarget
          const isOccupied = spot.isOccupied

          // Status colors
          const borderColor = isTarget
            ? '#06b6d4'
            : isOccupied
            ? '#f43f5e'
            : '#10b981'

          const bgColor = isTarget
            ? 'rgba(6, 182, 212, 0.16)'
            : isOccupied
            ? 'rgba(244, 63, 94, 0.12)'
            : 'rgba(16, 185, 129, 0.08)'

          return (
            <g
              key={spot.id}
              onClick={() => onToggleSpot(spot.id)}
              className="cursor-pointer group"
              style={{ transition: 'all 0.2s' }}
            >
              {/* Parking Stall Asphalt Box */}
              <rect
                x={spot.x - STALL_W / 2}
                y={spot.y - STALL_H / 2}
                width={STALL_W}
                height={STALL_H}
                rx="6"
                fill={bgColor}
                stroke={borderColor}
                strokeWidth={isTarget ? 2.5 : 1.5}
                strokeDasharray={isTarget ? '5 3' : 'none'}
                filter={isTarget ? 'url(#neonGlow)' : 'none'}
              />

              {/* Parking Stall Back Curb Stop / Wheel Bumper */}
              <rect
                x={spot.aisleSide === 'left' ? spot.x - STALL_W / 2 + 4 : spot.x + STALL_W / 2 - 8}
                y={spot.y - 14}
                width="4"
                height="28"
                rx="2"
                fill="#475569"
                opacity="0.7"
              />

              {/* Occupied Vehicle Silhouette if occupied */}
              {isOccupied && (
                <g transform={`translate(${spot.x}, ${spot.y})`}>
                  {/* Parked Car Body */}
                  <rect
                    x="-20"
                    y="-11"
                    width="40"
                    height="22"
                    rx="5"
                    fill="#334155"
                    stroke="#e11d48"
                    strokeWidth="1"
                    opacity="0.85"
                  />
                  {/* Windshields */}
                  <rect x="-8" y="-8" width="16" height="16" rx="2" fill="#0f172a" opacity="0.8" />
                  {/* Taillights */}
                  <rect x="-21" y="-9" width="1.5" height="4" fill="#f43f5e" />
                  <rect x="-21" y="5" width="1.5" height="4" fill="#f43f5e" />
                </g>
              )}

              {/* Available Spot Symbol */}
              {!isOccupied && (
                <g transform={`translate(${spot.x}, ${spot.y - 2})`} opacity="0.75">
                  <circle cx="0" cy="0" r="10" fill="none" stroke="#10b981" strokeWidth="1.2" />
                  <text
                    x="0"
                    y="3.5"
                    textAnchor="middle"
                    fill="#10b981"
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="sans-serif"
                  >
                    P
                  </text>
                </g>
              )}

              {/* Stall Name Tag */}
              <text
                x={spot.x}
                y={spot.y + (isOccupied ? 17 : 17)}
                textAnchor="middle"
                fill={isTarget ? '#22d3ee' : isOccupied ? '#fda4af' : '#6ee7b7'}
                fontSize="9.5"
                fontWeight="bold"
                fontFamily="monospace"
                letterSpacing="0.5px"
              >
                {spot.id}
              </text>

              {/* Target Banner for Selected Destination */}
              {isTarget && (
                <g transform={`translate(${spot.x}, ${spot.y - STALL_H / 2 - 12})`}>
                  <rect
                    x="-26"
                    y="0"
                    width="52"
                    height="14"
                    rx="7"
                    fill="#0891b2"
                    stroke="#22d3ee"
                    strokeWidth="1"
                  />
                  <text
                    x="0"
                    y="10"
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize="8"
                    fontWeight="bold"
                    fontFamily="sans-serif"
                  >
                    TARGET
                  </text>
                </g>
              )}
            </g>
          )
        })}

        {/* ------------------------------------------------------------- */}
        {/* 5. Intersections & Graph Nodes (Subtle Markers) */}
        {/* ------------------------------------------------------------- */}
        {Array.from(nodeCoords.entries()).map(([nodeId, coord]) => {
          if (coord.type === 'parking') return null
          const isJunction = coord.type === 'intersection'
          const isEntry = coord.type === 'entry'
          const isExit = coord.type === 'exit'

          return (
            <g key={nodeId} opacity={isJunction || isEntry || isExit ? 0.9 : 0.4}>
              <circle
                cx={coord.x}
                cy={coord.y}
                r={isEntry || isExit ? 7 : isJunction ? 5 : 3}
                fill={isEntry ? '#22c55e' : isExit ? '#ef4444' : isJunction ? '#f59e0b' : '#64748b'}
                stroke="#0f172a"
                strokeWidth="1.5"
              />
              <text
                x={coord.x}
                y={coord.y + (isEntry || isExit ? 17 : 13)}
                textAnchor="middle"
                fill={isEntry ? '#4ade80' : isExit ? '#f87171' : '#94a3b8'}
                fontSize="8"
                fontWeight="bold"
                fontFamily="monospace"
              >
                {nodeId}
              </text>
            </g>
          )
        })}

        {/* ------------------------------------------------------------- */}
        {/* 6. Entry & Exit Security Gates & Boom Barriers */}
        {/* ------------------------------------------------------------- */}
        {/* Entry Gate (x=0, y=1) */}
        {(() => {
          const p = gridToCanvas(0, 1)
          return (
            <g transform={`translate(${p.x - 55}, ${p.y - 20})`}>
              {/* Security Kiosk */}
              <rect x="0" y="0" width="22" height="40" rx="4" fill="#1e293b" stroke="#334155" strokeWidth="1.5" />
              <rect x="3" y="4" width="16" height="12" rx="2" fill="#0284c7" opacity="0.6" />
              {/* Green Beacon */}
              <circle cx="11" cy="-4" r="3.5" fill="#22c55e" filter="drop-shadow(0 0 5px #22c55e)" />
              {/* Boom Barrier Arm */}
              <line x1="22" y1="20" x2="48" y2="20" stroke="#f8fafc" strokeWidth="3" strokeDasharray="5 3" />
              <text x="11" y="52" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="bold">ENTRY</text>
            </g>
          )
        })()}

        {/* Exit Gate (x=10, y=1) */}
        {(() => {
          const p = gridToCanvas(10, 1)
          return (
            <g transform={`translate(${p.x + 35}, ${p.y - 20})`}>
              {/* Security Kiosk */}
              <rect x="0" y="0" width="22" height="40" rx="4" fill="#1e293b" stroke="#334155" strokeWidth="1.5" />
              <rect x="3" y="4" width="16" height="12" rx="2" fill="#ef4444" opacity="0.6" />
              {/* Red Beacon */}
              <circle cx="11" cy="-4" r="3.5" fill="#ef4444" filter="drop-shadow(0 0 5px #ef4444)" />
              {/* Boom Barrier Arm */}
              <line x1="-24" y1="20" x2="0" y2="20" stroke="#f8fafc" strokeWidth="3" strokeDasharray="5 3" />
              <text x="11" y="52" textAnchor="middle" fill="#f87171" fontSize="8" fontWeight="bold">EXIT</text>
            </g>
          )
        })()}

        {/* ------------------------------------------------------------- */}
        {/* 7. Highlighted Planned DQN Navigation Route */}
        {/* ------------------------------------------------------------- */}
        {plannedPathD && (
          <g>
            {/* Soft Ambient Glow */}
            <path
              d={plannedPathD}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="10"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.3"
              filter="url(#neonGlow)"
            />
            {/* Core Animated Dash Trail */}
            <path
              d={plannedPathD}
              fill="none"
              stroke="#38bdf8"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="10 8"
              className="animate-route-pulse"
            >
              <animate
                attributeName="stroke-dashoffset"
                values="0;-36"
                dur="1.2s"
                repeatCount="indefinite"
              />
            </path>
          </g>
        )}

        {/* ------------------------------------------------------------- */}
        {/* 8. Moving Autonomous Vehicle Sprite */}
        {/* ------------------------------------------------------------- */}
        <VehicleSprite
          x={vehiclePos.x}
          y={vehiclePos.y}
          rotation={vehiclePos.rotation}
          isMoving={isMoving}
          isParked={state?.is_parked ?? false}
          scale={0.9}
        />
      </svg>
    </div>
  )
}
