// Keep the original interface for other components
export interface FeaturePoint {
  index: number;
  coordinates: [number, number];
  original_embedding: number[];
  description: string;
}

// Query result type
export interface QueryResult {
  text: string;
  coordinates: [number, number];
  nearestFeatures: NearestFeature[];
}

// Nearest feature type
export interface NearestFeature {
  feature_id: string;
  similarity: number;
  description: string;
  coordinates: [number, number];
}

// Cluster data type
export interface ClusterData {
  clusterCount: number;
  labels: number[];
  colors: string[];
  centers: [number, number][];
  topics: Record<string, string[]>;
  topicScores: Record<string, number[]>;
  clusterColors: Record<string, string>;
}

export interface SAEScatterResponse {
  data: {
    coordinates: [number, number][];
    indices: string[];
    descriptions: string[];
    hierarchical_clusters: Record<string, ClusterData>;
    query?: {
      text: string;
      coordinates: [number, number];
      nearest_features: NearestFeature[];
    };
  };
}
