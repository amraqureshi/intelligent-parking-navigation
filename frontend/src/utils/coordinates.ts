// Coordinate mapping constants for the parking lot grid
export const CANVAS_WIDTH = 1060
export const CANVAS_HEIGHT = 620
export const PADDING_X = 95
export const PADDING_Y = 85
export const X_STEP = 87
export const Y_STEP = 112.5

export function gridToCanvas(gx: number, gy: number): { x: number; y: number } {
  const x = PADDING_X + gx * X_STEP
  // Invert Y so gy=5 is top, gy=1 is bottom
  const y = PADDING_Y + (5 - gy) * Y_STEP
  return { x, y }
}
