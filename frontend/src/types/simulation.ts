export interface NodeData {
  id: string
  x: number
  y: number
  type: 'entry' | 'exit' | 'intersection' | 'lane' | 'parking'
  label: string
}

export interface EdgeData {
  source: string
  target: string
  action: number
  action_name: string
  weight: number
}

export interface FacilityLayout {
  nodes: NodeData[]
  edges: EdgeData[]
  parking_spots: string[]
  grid_bounds: {
    min_x: number
    max_x: number
    min_y: number
    max_y: number
  }
}

export interface ValidAction {
  action: number
  name: string
}

export interface SimulationState {
  current_node: string
  current_pos: { x: number; y: number }
  current_step: number
  max_steps: number
  total_reward: number
  occupied_spots: string[]
  available_spots: string[]
  total_spots: number
  occupied_count: number
  available_count: number
  occupancy_rate: number
  status_map: Record<string, 'occupied' | 'available'>
  route_history: string[]
  planned_route: string[]
  is_searching: boolean
  is_parked: boolean
  parked_spot: string | null
  target_spot: string | null
  nearest_available_spot: string | null
  nearest_spot_distance: number | null
  valid_actions: ValidAction[]
  last_action: string | null
  last_message: string
  model_loaded: boolean
}

export interface SearchTransition {
  step: number
  action: number
  action_name: string
  from_node: string
  to_node: string
  reward: number
  success: boolean
  message: string
}

export interface SearchResult {
  success: boolean
  parking_spot: string | null
  route: string[]
  steps: number
  total_reward: number
  transitions: SearchTransition[]
  algorithm: string
  message: string
}

export interface StepResult {
  action: number
  action_name: string
  prev_node: string
  current_node: string
  reward: number
  total_reward: number
  done: boolean
  success: boolean
  parking_spot: string | null
  message: string
  state: SimulationState
}

export interface ActivityEvent {
  id: string
  timestamp: string
  type: 'info' | 'success' | 'warning' | 'action' | 'dqn'
  title: string
  description: string
  node?: string
}

export interface ModelMetrics {
  model_architecture: {
    type: string
    framework: string
    state_dimension: number
    action_dimension: number
    layers: Array<{ name: string; units: number; activation?: string }>
    training_episodes: number
    loss_function: string
    target_update: string
  }
  benchmarks: {
    num_episodes_per_level: number
    occupancy_rates: number[]
    metrics_by_occupancy: Record<string, {
      baseline: {
        success_rate_pct: number
        avg_steps: number
        avg_reward: number
        invalid_moves_total: number
      }
      dqn: {
        success_rate_pct: number
        avg_steps: number
        avg_reward: number
        invalid_moves_total: number
      }
    }>
  } | null
  facility_specs: {
    total_nodes: number
    total_parking_spots: number
    driving_lanes: string[]
    grid_dimensions: string
  }
}
