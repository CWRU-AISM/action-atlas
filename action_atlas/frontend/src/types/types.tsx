"use client";
import { Chart as ChartJS, TooltipItem } from "chart.js";
import React from "react";
import { NearestFeature } from "@/types/sae";
import { FeatureDetailResponse } from "@/types/feature";

export interface QueryResult {
  bins: number[];
  counts: number[];
}

// 用于存储左侧 Query search 表单的结构体,@/components/Concept Query
// 一个绘制图像的结构体,@/components/Concept Query
export interface PlotProps {
  queryResult: { bins: number[]; counts: number[] };
  chartRef: React.RefObject<ChartJS<"line">>;
}

// @/components/VisualizationDashboard 下拉框选项
export interface AttributeSelectProps {
  selectedAttrs: string[];
  setSelectedAttrs: React.Dispatch<React.SetStateAction<string[]>>;
  metricOptions: { value: string; label: string }[];
}

// @/components/VisualizationDashboard/index 假数据
export interface ModelData {
  id: string;
  type: string;
  layer: number;
  l0_sparsity?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  l2_ratio?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  explained_variance?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  kl_div_score?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  ce_loss_score?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  llm_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  llm_top_1_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  llm_top_2_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  llm_top_5_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  sae_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  sae_top_1_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  sae_top_2_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  sae_top_5_test_accuracy?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  scr_metric_threshold_10?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  scr_metric_threshold_20?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  tpp_threshold_10?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  tpp_threshold_20?: { value: number | null; rank: number | null } | { value: 0; rank: 26 };
  top_10_score?: { value: number; rank: number | null } | { value: 0; rank: 26 };
  top_100_score?: { value: number; rank: number | null } | { value: 0; rank: 26 };
  top_1000_score?: { value: number; rank: number | null } | { value: 0; rank: 26 };

  // avgRank: number;
  [key: string]:
    | string
    | number
    | { value: number | null; rank: number | null }
    | { value: 0; rank: 26 }
    | undefined;
}

// @components/VisualizationDashBoard 可视化矩阵组件
export interface VisibleRange {
  start: number;
  end: number;
}

// @components/VisualizationDashBoard 可视化矩阵组件
export interface OverviewMatrixProps {
  modelData: ModelData[];
  visibleRange: VisibleRange;
  calculateAverageRank: (model: ModelData) => number;
}

export interface ModelTableProps {
  groupedData: ModelData[][];
  selectedModel: ModelData | null;
  setSelectedModel: (model: ModelData) => void;
  visibleRange: VisibleRange;
  setVisibleRange: (range: VisibleRange) => void;
  getTopContributingAttrs: (
    model: ModelData
  ) => { attr: string; contribution: number }[];
  calculateAverageRank: (model: ModelData) => number;
}

export interface RadarPlotChartProps {
  modelData: ModelData[];
  selectedModel: ModelData | null;
  calculateAverage: (attr: string) => number;
  selectedAttrs: string[];
}

export interface ModelRadarPanelProps {
  selectedModel: ModelData | null;
  // modelData: ModelData[];
  // selectedAttrs: string[];
  // calculateAverage: (attr: string) => number;
}

// explanation Feature mock data
export interface TokenStats {
  neg_tokens: {
    tokens: string[];
    values: number[];
  };
  pos_tokens: {
    tokens: string[];
    values: number[];
  };
  freq_histogram: {
    heights: number[];
    values: number[];
  };
  logits_histogram: {
    heights: number[];
    values: number[];
  };
}

// ActivationData 类型
export interface ActivationData {
  sentence: string;
  similarity: number;
  max_value: number;
  max_value_token: string;
  maxValueTokenIndex: number;
  token_value_pairs: { token: string; value: number }[];
  Interval_Bin: [number, number];
  Interval_contains: boolean;
}

export interface SelectedToken {
  token: string;
  relative_activation: number;
}

export interface FeatureSectionProps {
  featureInfo?: { data: FeatureDetailResponse };
}

export interface Point {
  x: number;
  y: number;
  bin?: number;
  id: string;
  // radius: number; // 新增属性
  sentence: { token: string; value: number }[];
  keyword: string;
  similarity: number;
}

export interface ScatterPoint {
  x: number;
  y: number;
  index: number;
  featureId: number;
  clusterId: number;
  description: string;
  layer?: string; // Layer identifier (e.g., "12" from "action_expert_layer_12")
  isQuery?: boolean;
  isQuerySimilar?: boolean;
  isSelected?: boolean;
  isSelectedSimilar?: boolean;
  isConceptHighlighted?: boolean; // Set when a concept is selected and this feature matches
  isSearchHighlighted?: boolean; // Set when a feature matches search/filter criteria
  color?: string;
  visible?: boolean;
  similarity?: number;
  relatedToken?: Array<{
    prompt: string;
    index: number;
    activation: number;
  }> | null;
  isVisibleInPanel?: boolean;
}

export interface SelectionBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface Coordinates {
  x: number;
  y: number;
}

export interface MatrixPlotProps {
  onPointsSelected: (points: Point[]) => void;
}

export interface InputPromptProps {
  prompt: string;
  setPrompt: (value: string) => void;
  handleSteer: () => void;
  loading: boolean;
}

export interface SteerOutput {
  strength: number;
  model_output: string;
  embedding: number[];
  similarity_to_explanation: number;
  similarity_to_default: number;
}

export interface TreeNode {
  id: string;
  value: string | number;
  children?: TreeNode[];
}

export interface TreeViewProps {
  strengths: number[];
  setStrengthsAction: (strengths: number[]) => void;
  prompt: string;
  results: SteerOutput[];
  setResultsAction: (results: SteerOutput[]) => void;
  selectedBranch: number | null;
  setSelectedBranchAction: (branch: number | null) => void;
}

export interface QueryInputProps {
  onSubmit: (query: string, llm: string) => Promise<void>; // 修改这行，添加 llm 参数
}

export const chartData = (queryResult: PlotProps["queryResult"]) => ({
  labels: queryResult.bins.map((bin) => bin.toFixed(2)),
  datasets: [
    {
      label: "Count",
      data: queryResult.counts,
      borderColor: "#1976d1",
      borderWidth: 2,
      tension: 0.4,
      fill: true,
      pointRadius: 3,
      pointBackgroundColor: "#1976d1",
      pointBorderColor: "#fff",
      pointBorderWidth: 2,
    },
  ],
});

export const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: { display: false },
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (context: TooltipItem<"line">) =>
          `Similarity: ${context.label}, Count: ${context.parsed.y}`,
      },
      backgroundColor: "rgba(0, 0, 0, 0.8)",
      titleColor: "#fff",
      bodyColor: "#fff",
      padding: 8,
      displayColors: false,
    },
  },
  scales: {
    x: {
      title: {
        display: false,
      },
      grid: { display: false },
      ticks: {
        font: {
          size: 10,
        },
      },
    },
    y: {
      title: {
        display: false,
      },
      beginAtZero: true,
      grid: { color: "rgba(0, 0, 0, 0.1)" },
      ticks: {
        font: {
          size: 10,
        },
      },
    },
  },
};

export interface SimilarFeaturesPanelProps {
  similarFeatures: NearestFeature[];
  selectedFeatureId: string | null;
  hasQueryPoint: boolean;
  onFeatureClick: (feature: NearestFeature) => void;
  onVisibleFeaturesChange?: (featureIds: string[]) => void;
}

// 添加与父组件相同的 FeatureInfo 接口
export interface FeatureInfo {
  explanation: string;
  cosine_similarity: number;
}

// 单个特征的激活信息
export interface FeatureActivation {
  feature_id: string;
  activation: number;
}

// 每个令牌相关的特征信息
export interface TokenFeatureInfo {
  prompt: string;
  token_index: number;
  features: FeatureActivation[];
}

// 令牌分析请求参数
export interface TokenAnalysisRequest {
  feature_id: string;
  sae_id: string;
  selected_prompt_tokens: {
    prompt: string;
    token_index: number;
  }[];
}

// 将空接口替换为完整的定义
export interface TokenAnalysisResults {
  original_explanation: string;
  related_features_intersection: string[];
  related_features_union: string[];
  prompt_token_features: {
    [key: string]: TokenFeatureInfo; // 键的格式为: "prompt_tokenIndex"
  };
}
