import type {
  FacilityLayout,
  SimulationState,
  SearchResult,
  StepResult,
  ModelMetrics,
} from '../types/simulation'

const API_BASE = import.meta.env.VITE_API_URL || ''

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    let errorDetail = `Request failed: ${response.status} ${response.statusText}`
    try {
      const errorJson = await response.json()
      if (errorJson.detail) {
        errorDetail = errorJson.detail
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail)
  }

  return response.json()
}

export const api = {
  // Health check
  getHealth: () => request<{
    status: string
    service: string
    model_loaded: boolean
    checkpoint_exists: boolean
    checkpoint_path: string
    device: string
    num_nodes: number
    num_spots: number
  }>('/api/health'),

  // Facility topology
  getLayout: () => request<FacilityLayout>('/api/layout'),

  // Current simulation state
  getState: () => request<SimulationState>('/api/state'),

  // Reset simulation
  reset: (params?: {
    start_node?: string
    occupied_spots?: string[]
    random_occupancy_rate?: number
    seed?: number
  }) =>
    request<SimulationState>('/api/reset', {
      method: 'POST',
      body: JSON.stringify(params || {}),
    }),

  // Toggle parking spot occupancy
  toggleSpot: (spotId: string, occupied?: boolean) =>
    request<SimulationState>('/api/toggle-spot', {
      method: 'POST',
      body: JSON.stringify({ spot_id: spotId, occupied }),
    }),

  // Set scenario preset
  setScenario: (preset: string) =>
    request<SimulationState>('/api/scenario', {
      method: 'POST',
      body: JSON.stringify({ preset }),
    }),

  // Execute DQN search
  startSearch: () =>
    request<SearchResult>('/api/search', {
      method: 'POST',
    }),

  // Single step in environment
  step: () =>
    request<StepResult>('/api/step', {
      method: 'POST',
    }),

  // Evaluation & architecture metrics
  getMetrics: () => request<ModelMetrics>('/api/metrics'),
}
