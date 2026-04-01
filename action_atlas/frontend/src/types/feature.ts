export interface FeatureDetailResponse {
  feature_info: {
    feature_id: string;
    sae_id: string;
    layer: number;
    type: string;
    index: number;
  };
  activation_data: Array<{
    sentence: string;
    similarity: number;
    similarity_rank: number;
    max_value: number;
    max_value_rank: number;
    max_value_token: string;
    maxValueTokenIndex: number;
    token_value_pairs: Array<{ token: string; value: number }>;
    Interval_Bin: [number, number];
    Interval_contains: string;
  }>;
  explanation: string;
  raw_stats: {
    neg_tokens: { tokens: string[]; values: number[] };
    pos_tokens: { tokens: string[]; values: number[] };
    freq_histogram: { heights: number[]; values: number[] };
    logits_histogram: { heights: number[]; values: number[] };
    similar_features: { feature_ids: number[]; values: number[]; };
  };
  bins_statistics: {
    bins_data: Array<{
      bin_range: [number, number];
      sample_count: number;
      bin_contains: string;
    }>;
    total_samples: number;
  };
}

export interface TokenData {
  token: string;
  activation_value: number;
  relative_activation: number;
}

export interface PromptResult {
  prompt: string;
  tokens: TokenData[];
  max_value: number;
  max_value_token_index: number;
  min_value: number;
}

export interface TokenActivationRequest {
  feature_id: string;
  prompts: string[];
  sae_id: string;
}

export interface TokenActivationResponse {
  data: {
    combined_response_path: string;
    prompt_results: PromptResult[];
    request: TokenActivationRequest;
  };
  status: number;
}

export interface TokenAnalysisRequest {
  feature_id: string;
  sae_id: string;
  selected_tokens: {
    token: string;
    relative_activation: number;
  }[];
}

export interface TokenAnalysisResponse {
  status: number;
  data: {
    original_explanation: string;
    related_features_intersection: string[];
    related_features_union: string[];
    token_features: {
      [key: string]: {
        feature_id: string;
        activation: number;
      }[];
    };
  };
}
