// 保留原有接口用于其他组件
export interface FeaturePoint {
  index: number;
  coordinates: [number, number];
  original_embedding: number[];
  description: string;
}

// 查询结果类型
export interface QueryResult {
  text: string;
  coordinates: [number, number];
  nearestFeatures: NearestFeature[];
}

// 最近特征类型
export interface NearestFeature {
  feature_id: string;
  similarity: number;
  description: string;
  coordinates: [number, number];
}

// 聚类数据类型
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
