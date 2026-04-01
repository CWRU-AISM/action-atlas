"use client";
import { Paper, Skeleton } from "@mui/material";
import React, { useState, useRef } from "react";
import dynamic from "next/dynamic";
import QueryInput from "@/app/components/Concept Query/QueryInput";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import {
  setLayerTypeData,
  setQueryResult,
  setOptimizedQuery,
  setOptimizedQueryResult,
} from "@/redux/features/querySlice";
import { setModelData } from "@/redux/features/modelSlice";
import { API_BASE_URL } from "@/config/api";
// VLA model - no static metrics data needed

const Plot = dynamic(() => import("@/app/components/Concept Query/Plot"), { ssr: false });

export default function ConceptQuery() {
  const dispatch = useAppDispatch();
  const queryResult = useAppSelector((state) => state.query.queryResult);
  const modelData = useAppSelector((state) => state.model.modelData);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [timeoutError, setTimeoutError] = useState(false);

  const handleSearch = async (query: string, llm: string) => {
    if (!query) return;

    setError(null);
    setTimeoutError(false);
    setLoading(true);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        setTimeoutError(true);
        setLoading(false);
      }, 60 * 1000);

      const response = await fetch(`${API_BASE_URL}/api/query/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: query.trim(),
          llm: llm
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        setError(errorData.message || `请求失败: ${response.status}`);
        setLoading(false);
        return;
      }

      const data = await response.json();
      
      // Model configuration based on model type
      const getModelConfig = (model: string) => {
        switch (model) {
          case 'pi05':
          default:
            return {
              layers: 18,
              idFormat: (layer: number) => `action_expert_layer_${layer}-concepts`
            };
        }
      };

      const modelConfig = getModelConfig(llm);

      dispatch(setQueryResult({
        bins: data.data.similarity_distribution.bins,
        counts: data.data.similarity_distribution.counts,
      }));

      if (data.data.similarity_distribution_optimized) {
        dispatch(setOptimizedQuery(data.data.similarity_distribution_optimized.query));
        dispatch(setOptimizedQueryResult({
          bins: data.data.similarity_distribution_optimized.bins,
          counts: data.data.similarity_distribution_optimized.counts,
        }));
      }

      dispatch(setLayerTypeData({
        features: data.data.features,
        saeDistributions: data.data.sae_distributions,
      }));

      if (data.data.sae_distributions) {
        // 根据模型配置生成modelData
        const updatedModelData = Array.from({ length: modelConfig.layers }, (_, layer) => {
          const modelId = modelConfig.idFormat(layer);
          
          const distributionData = {
            top_10: data.data.sae_distributions.top_10?.distribution[modelId] || { percentage: 0, count: modelConfig.layers },
            top_100: data.data.sae_distributions.top_100?.distribution[modelId] || { percentage: 0, count: modelConfig.layers },
            top_1000: data.data.sae_distributions.top_1000?.distribution[modelId] || { percentage: 0, count: modelConfig.layers },
          };

          // 基础模型数据
          const baseModel = {
            id: modelId,
            type: "RES" as const,
            layer: layer,
            top_10_score: { value: distributionData.top_10.percentage, rank: distributionData.top_10.count },
            top_100_score: { value: distributionData.top_100.percentage, rank: distributionData.top_100.count },
            top_1000_score: { value: distributionData.top_1000.percentage, rank: distributionData.top_1000.count },
          };

          // 默认属性列表
          const defaultAttributes = {
            l0_sparsity: { value: 0, rank: modelConfig.layers },
            l2_ratio: { value: 0, rank: modelConfig.layers },
            explained_variance: { value: 0, rank: modelConfig.layers },
            kl_div_score: { value: 0, rank: modelConfig.layers },
            ce_loss_score: { value: 0, rank: modelConfig.layers },
            llm_test_accuracy: { value: 0, rank: modelConfig.layers },
            llm_top_1_test_accuracy: { value: 0, rank: modelConfig.layers },
            llm_top_2_test_accuracy: { value: 0, rank: modelConfig.layers },
            llm_top_5_test_accuracy: { value: 0, rank: modelConfig.layers },
            sae_test_accuracy: { value: 0, rank: modelConfig.layers },
            sae_top_1_test_accuracy: { value: 0, rank: modelConfig.layers },
            sae_top_2_test_accuracy: { value: 0, rank: modelConfig.layers },
            sae_top_5_test_accuracy: { value: 0, rank: modelConfig.layers },
            scr_metric_threshold_10: { value: 0, rank: modelConfig.layers },
            scr_metric_threshold_20: { value: 0, rank: modelConfig.layers },
            tpp_threshold_10: { value: 0, rank: modelConfig.layers },
            tpp_threshold_20: { value: 0, rank: modelConfig.layers },
          };

          // Merge all data
          return {
            ...baseModel,
            ...defaultAttributes
          };
        });

        dispatch(setModelData(updatedModelData));
      }

      setLoading(false);
    } catch (error) {
      if (!timeoutError) setError("Search request failed, please try again later.");
      setLoading(false);
    }
  };

  const chartRef = useRef<any>(null);

  return (
    <Paper className="overflow-hidden w-full h-full flex flex-col rounded-lg shadow-md">
      {/* Dark Navy Header */}
      <div className="h-8 flex items-center px-3 bg-[#0a1628] rounded-t-lg">
        <span className="text-xs font-semibold text-white">Concept Query</span>
      </div>
      <div className="flex-1 p-3 overflow-hidden flex flex-col bg-white">
        <QueryInput onSubmit={handleSearch} />
        {error && <div className="text-red-500 text-xs my-1">{error}</div>}
        {timeoutError && <div className="text-red-500 text-xs my-1">Request timed out, please try again later.</div>}
        {/* Similarity distribution plot hidden for VLA - not relevant for robot features */}
        {loading && (
          <Skeleton animation="wave" variant="rectangular" className="w-full h-8 mt-2" />
        )}
      </div>
    </Paper>
  );
}