import React from 'react'

interface VehicleSpriteProps {
  x: number
  y: number
  rotation: number
  isMoving: boolean
  isParked: boolean
  scale?: number
}

export const VehicleSprite: React.FC<VehicleSpriteProps> = ({
  x,
  y,
  rotation,
  isMoving,
  isParked,
  scale = 1.0,
}) => {
  return (
    <g
      transform={`translate(${x}, ${y}) rotate(${rotation}) scale(${scale})`}
      style={{
        transition: 'transform 0.15s cubic-bezier(0.25, 0.1, 0.25, 1)',
        willChange: 'transform',
      }}
      className="vehicle-group"
    >
      {/* Dynamic Headlight Cones / Beams (Cast on Road ahead) */}
      {!isParked && (
        <g opacity={isMoving ? 0.75 : 0.5}>
          {/* Left Headlight Beam */}
          <polygon
            points="24,-8 90,-28 90,8 24,-2"
            fill="url(#headlightGradient)"
            opacity="0.6"
          />
          {/* Right Headlight Beam */}
          <polygon
            points="24,8 90,-8 90,28 24,2"
            fill="url(#headlightGradient)"
            opacity="0.6"
          />
        </g>
      )}

      {/* Vehicle Drop Shadow */}
      <rect
        x="-24"
        y="-14"
        width="48"
        height="28"
        rx="8"
        fill="#000000"
        opacity="0.5"
        transform="translate(2, 4)"
        filter="blur(3px)"
      />

      {/* Main Car Body Chassis */}
      <rect
        x="-24"
        y="-13"
        width="48"
        height="26"
        rx="7"
        fill="url(#carBodyGradient)"
        stroke={isParked ? '#10b981' : '#38bdf8'}
        strokeWidth="1.5"
      />

      {/* Wheels */}
      {/* Front Left Wheel */}
      <rect x="10" y="-15" width="9" height="3" rx="1.5" fill="#1e293b" stroke="#0f172a" />
      {/* Front Right Wheel */}
      <rect x="10" y="12" width="9" height="3" rx="1.5" fill="#1e293b" stroke="#0f172a" />
      {/* Rear Left Wheel */}
      <rect x="-19" y="-15" width="9" height="3" rx="1.5" fill="#1e293b" stroke="#0f172a" />
      {/* Rear Right Wheel */}
      <rect x="-19" y="12" width="9" height="3" rx="1.5" fill="#1e293b" stroke="#0f172a" />

      {/* Hood Grooves */}
      <line x1="8" y1="-8" x2="19" y2="-6" stroke="#0284c7" strokeWidth="1" opacity="0.6" />
      <line x1="8" y1="8" x2="19" y2="6" stroke="#0284c7" strokeWidth="1" opacity="0.6" />

      {/* Front Windshield */}
      <path
        d="M 5,-10 L 11,-8 L 11,8 L 5,10 Z"
        fill="#0f172a"
        stroke="#1e293b"
        strokeWidth="0.8"
      />

      {/* Cabin / Panoramic Glass Sunroof */}
      <rect
        x="-13"
        y="-9"
        width="19"
        height="18"
        rx="3"
        fill="#0f172a"
        stroke="#334155"
        strokeWidth="0.8"
      />

      {/* Sunroof Interior Glint */}
      <rect
        x="-11"
        y="-7"
        width="15"
        height="14"
        rx="2"
        fill="url(#glassGradient)"
        opacity="0.7"
      />

      {/* Rear Window */}
      <path
        d="M -15,-9 L -19,-7 L -19,7 L -15,9 Z"
        fill="#0f172a"
        stroke="#1e293b"
        strokeWidth="0.8"
      />

      {/* Side Mirrors */}
      <ellipse cx="8" cy="-14" rx="2" ry="1.5" fill="#0284c7" />
      <ellipse cx="8" cy="14" rx="2" ry="1.5" fill="#0284c7" />

      {/* LED Headlights */}
      <ellipse cx="23" cy="-9" rx="2.5" ry="2" fill="#e0f2fe" filter="drop-shadow(0 0 4px #38bdf8)" />
      <ellipse cx="23" cy="9" rx="2.5" ry="2" fill="#e0f2fe" filter="drop-shadow(0 0 4px #38bdf8)" />

      {/* LED Taillights (Red) */}
      <rect
        x="-24.5"
        y="-11"
        width="1.8"
        height="5"
        rx="0.9"
        fill={isParked ? '#10b981' : '#ef4444'}
        filter={isParked ? 'drop-shadow(0 0 3px #10b981)' : 'drop-shadow(0 0 3px #ef4444)'}
      />
      <rect
        x="-24.5"
        y="6"
        width="1.8"
        height="5"
        rx="0.9"
        fill={isParked ? '#10b981' : '#ef4444'}
        filter={isParked ? 'drop-shadow(0 0 3px #10b981)' : 'drop-shadow(0 0 3px #ef4444)'}
      />

      {/* Status Glow Indicator on Car Roof */}
      <circle
        cx="-3"
        cy="0"
        r="3"
        fill={isParked ? '#10b981' : isMoving ? '#38bdf8' : '#f59e0b'}
      >
        {isMoving && (
          <animate
            attributeName="r"
            values="2.5;4;2.5"
            dur="1.2s"
            repeatCount="indefinite"
          />
        )}
      </circle>
    </g>
  )
}
