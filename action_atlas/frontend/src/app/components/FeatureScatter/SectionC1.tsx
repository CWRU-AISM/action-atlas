"use client";
import { Paper, Typography, Box, Chip, TextField, InputAdornment, IconButton, Select, MenuItem, FormControl, InputLabel, Tooltip } from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import ClearIcon from "@mui/icons-material/Clear";
import React, { useCallback, useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { ClusterData, NearestFeature, SAEScatterResponse } from "@/types/sae";
import SimilarFeaturesPanel from "./SimilarFeaturesPanel";
import FeatureDetailPanel from "./FeatureDetailPanel";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import {
  setSelectedFeature,
  setSelectedTokens,
  setValidateFeatureId,
} from "@/redux/features/featureSlice";
import { clearSelectedConcept } from "@/redux/features/conceptSlice";
import {
  FeatureActivation,
  ScatterPoint,
  TokenAnalysisResults,
  TokenFeatureInfo,
} from "@/types/types";

// Concept identification method options
type ConceptMethod = 'contrastive' | 'ffn';
// Pi0.5 pathway options
type Pi05Pathway = 'expert' | 'paligemma';

import {
  convertClusterData,
  createMainPoints,
  createQueryPoint,
  initD3Chart,
  initZoom,
  updateHexbins,
  LAYER_COLORS,
  CONCEPT_TYPE_COLORS,
  CONCEPT_TYPE_MAP,
  extractPrimaryConcept,
  ColoringMode,
} from "@/utils/utils";
import { FeatureDetailResponse } from "@/types/feature";
import { debounce } from "lodash";
import { API_BASE_URL, apiUrl } from "@/config/api";

// Concept category options for filtering
const CONCEPT_CATEGORIES = [
  { value: "all", label: "All Categories" },
  { value: "motion", label: "Motion" },
  { value: "object", label: "Object" },
  { value: "spatial", label: "Spatial" },
  { value: "phase", label: "Action Phase" },
] as const;

type ConceptCategory = typeof CONCEPT_CATEGORIES[number]["value"];

interface SearchResult {
  featureId: number;
  description: string;
  conceptType: string;
  point: ScatterPoint;
  similarity?: number;
  searchMethod?: "semantic" | "text";
}

interface SemanticSearchResult {
  feature_id: number;
  description: string;
  layer: string;
  suite: string;
  similarity: number;
  pathway?: string;
}

export default function SectionC1() {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const dataRef = React.useRef<ScatterPoint[]>([]);
  const selectedClusterRef = React.useRef<{ level: string; data: ClusterData }>(
    {
      level: "10",
      data: {
        clusterCount: 10,
        labels: [] as number[],
        colors: [] as string[],
        centers: [] as [number, number][],
        topics: {} as Record<string, string[]>,
        topicScores: {} as Record<string, number[]>,
        clusterColors: {} as Record<string, string>,
      },
    }
  );
  const [similarFeatures, setSimilarFeatures] = React.useState<
    NearestFeature[]
  >([]);
  const [, setClusterLevels] = React.useState<string[]>([]);
  const [hierarchicalClusters, setHierarchicalClusters] = React.useState<
    SAEScatterResponse["data"]["hierarchical_clusters"]
  >({});
  const [selectedFeatureId, setSelectedFeatureId] = React.useState<
    string | null
  >(null);
  const zoomRef = React.useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>(null!);
  const xScale = React.useRef<d3.ScaleLinear<number, number>>(null!);
  const yScale = React.useRef<d3.ScaleLinear<number, number>>(null!);
  const [queryInfo, setQueryInfo] = React.useState<
    SAEScatterResponse["data"]["query"] | null
  >(null);
  const zoomThresholds = {
    low: { zoom: 1, level: "10" },
    medium: { zoom: 3, level: "30" },
    high: { zoom: 7, level: "90" },
  };
  const tooltipRef = React.useRef<
    d3.Selection<HTMLDivElement, unknown, HTMLElement, any>
  >(null!);
  const transformRef = React.useRef<d3.ZoomTransform>(d3.zoomIdentity);
  const [featureDetails, setFeatureDetails] = React.useState<NearestFeature[]>(
    []
  );
  const [, setSelectedPointId] = React.useState<number | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);
  const [coloringMode, setColoringMode] = React.useState<"cluster" | "concept" | "layer">("cluster");

  // Search and filter state
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [categoryFilter, setCategoryFilter] = useState<ConceptCategory>("all");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearchPanelOpen, setIsSearchPanelOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [autocompleteSuggestions, setAutocompleteSuggestions] = useState<string[]>([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [searchMethod, setSearchMethod] = useState<"semantic" | "text">("text");
  const [isSemanticSearching, setIsSemanticSearching] = useState(false);
  const autocompleteRef = useRef<HTMLDivElement>(null);

  // Method and pathway state for concept identification
  const [conceptMethod, setConceptMethod] = useState<ConceptMethod>("contrastive");
  const [pi05Pathway, setPi05Pathway] = useState<Pi05Pathway>("expert");

  const { sae } = useAppSelector((state) => state.sae);
  // For VLA: Use selectedLLM which contains "layer-suite" format as the SAE ID
  const selectedLLM = useAppSelector(state => state.llm.selectedLLM);
  const vlaSaeId = selectedLLM || sae;  // Prefer VLA layer-suite, fallback to sae
  console.log("SAE ID:", vlaSaeId);
  const { query } = useAppSelector((state) => state.query);
  const currentLLM = useAppSelector(state => state.query.currentLLM);
  const currentModel = useAppSelector(state => state.model.currentModel);

  // When switching models, reset method/pathway to valid defaults
  React.useEffect(() => {
    if (currentModel !== 'pi05') {
      // FFN and paligemma are Pi0.5-only; force contrastive+expert for other models
      setConceptMethod('contrastive');
      setPi05Pathway('expert');
    }
  }, [currentModel]);

  // Concept highlighting
  const { selectedConcept, selectedType, highlightedFeatureIds } = useAppSelector((state) => state.concept);

  const dispatch = useAppDispatch();

  const [visiblePanelFeatures, setVisiblePanelFeatures] = useState<string[]>(
    []
  );

  // Search and filter functions
  const performSearch = useCallback((query: string, category: ConceptCategory) => {
    if (!dataRef.current.length) {
      setSearchResults([]);
      return;
    }

    const queryLower = query.toLowerCase().trim();

    const results = dataRef.current
      .filter((point) => {
        if (point.isQuery) return false;

        // Get concept type for this point
        const concept = extractPrimaryConcept(point.description || "");
        const conceptType = concept ? CONCEPT_TYPE_MAP[concept] : "unknown";

        // Apply category filter
        if (category !== "all" && conceptType !== category) {
          return false;
        }

        // Apply text search filter
        if (queryLower && point.description) {
          return point.description.toLowerCase().includes(queryLower);
        }

        // If no text query but category filter is active, include all matching category
        return category !== "all" || queryLower !== "";
      })
      .map((point) => {
        const concept = extractPrimaryConcept(point.description || "");
        const conceptType = concept ? CONCEPT_TYPE_MAP[concept] : "unknown";
        return {
          featureId: point.featureId,
          description: point.description,
          conceptType,
          point,
        };
      })
      .slice(0, 500); // Show more results (up to 500)

    setSearchResults(results);

    // Update highlight state for search results
    const searchMatchIds = new Set(results.map(r => r.featureId));
    dataRef.current = dataRef.current.map((point) => ({
      ...point,
      isSearchHighlighted: searchMatchIds.has(point.featureId),
    }));

    // Redraw to show highlights
    if (svgRef.current) {
      handleUpdateHexbins(transformRef.current.k);
    }
  }, []);

  // Semantic search via API with fallback to client-side text search
  const performSemanticSearch = useCallback(async (query: string, category: ConceptCategory) => {
    if (!query.trim()) {
      // No query text: fall back to client-side category filter only
      performSearch(query, category);
      setSearchMethod("text");
      setAutocompleteSuggestions([]);
      setShowAutocomplete(false);
      return;
    }

    setIsSemanticSearching(true);
    try {
      const url = new URL(`${API_BASE_URL}/api/vla/semantic_search`, window.location.origin);
      url.searchParams.append("q", query.trim());
      if (currentModel) url.searchParams.append("model", currentModel);
      url.searchParams.append("limit", "20");

      const response = await fetch(url.toString());
      if (!response.ok) throw new Error(`Semantic search failed: ${response.status}`);

      const data = await response.json();
      const apiResults: SemanticSearchResult[] = data.results || [];
      const suggestions: string[] = data.autocomplete || [];

      if (apiResults.length === 0) {
        // No semantic results: fall back to client-side text search
        performSearch(query, category);
        setSearchMethod("text");
        setAutocompleteSuggestions(suggestions);
        setShowAutocomplete(suggestions.length > 0);
        setIsSemanticSearching(false);
        return;
      }

      // Map API results to SearchResult format, matching against scatter data points
      const semanticResults: SearchResult[] = apiResults
        .filter((apiResult) => {
          if (category === "all") return true;
          const concept = extractPrimaryConcept(apiResult.description || "");
          const conceptType = concept ? CONCEPT_TYPE_MAP[concept] : "unknown";
          return conceptType === category;
        })
        .map((apiResult) => {
          const matchingPoint = dataRef.current.find(
            (p) => p.featureId === apiResult.feature_id
          );
          const concept = extractPrimaryConcept(apiResult.description || "");
          const conceptType = concept ? CONCEPT_TYPE_MAP[concept] : "unknown";

          return {
            featureId: apiResult.feature_id,
            description: apiResult.description,
            conceptType,
            similarity: apiResult.similarity,
            searchMethod: "semantic" as const,
            point: matchingPoint || {
              x: 0, y: 0, index: 0,
              featureId: apiResult.feature_id,
              clusterId: 0,
              description: apiResult.description,
            },
          };
        });

      setSearchResults(semanticResults);
      setSearchMethod("semantic");
      setAutocompleteSuggestions(suggestions);
      setShowAutocomplete(suggestions.length > 0);

      // Highlight matching features on the scatter plot
      const searchMatchIds = new Set(semanticResults.map((r) => r.featureId));
      dataRef.current = dataRef.current.map((point) => ({
        ...point,
        isSearchHighlighted: searchMatchIds.has(point.featureId),
      }));

      if (svgRef.current) {
        handleUpdateHexbins(transformRef.current.k);
      }
    } catch (error) {
      console.warn("Semantic search failed, falling back to text search:", error);
      performSearch(query, category);
      setSearchMethod("text");
    } finally {
      setIsSemanticSearching(false);
    }
  }, [currentModel, performSearch]);

  // Debounced search - calls semantic API with text fallback
  const debouncedSearch = useCallback(
    debounce((query: string, category: ConceptCategory) => {
      performSemanticSearch(query, category);
    }, 300),
    [performSemanticSearch]
  );

  // Handle search input change
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = event.target.value;
    setSearchQuery(newQuery);
    setIsSearchPanelOpen(true);
    setShowAutocomplete(true);
    debouncedSearch(newQuery, categoryFilter);
  };

  // Handle autocomplete suggestion click
  const handleAutocompleteSuggestionClick = (suggestion: string) => {
    setSearchQuery(suggestion);
    setShowAutocomplete(false);
    setIsSearchPanelOpen(true);
    debouncedSearch.cancel();
    performSemanticSearch(suggestion, categoryFilter);
  };

  // Handle category filter change
  const handleCategoryChange = (event: any) => {
    const newCategory = event.target.value as ConceptCategory;
    setCategoryFilter(newCategory);
    setIsSearchPanelOpen(true);
    debouncedSearch(searchQuery, newCategory);
  };

  // Clear search
  const handleClearSearch = () => {
    setSearchQuery("");
    setCategoryFilter("all");
    setSearchResults([]);
    setIsSearchPanelOpen(false);
    setAutocompleteSuggestions([]);
    setShowAutocomplete(false);
    setSearchMethod("text");

    // Clear search highlights
    dataRef.current = dataRef.current.map((point) => ({
      ...point,
      isSearchHighlighted: false,
    }));

    if (svgRef.current) {
      handleUpdateHexbins(transformRef.current.k);
    }
  };

  // Handle clicking on a search result
  const handleSearchResultClick = (result: SearchResult) => {
    zoomToPoint(result.point, 6);
    setSelectedFeatureId(result.featureId.toString());

    // Fetch feature details
    fetch(
      `${API_BASE_URL}/api/feature/detail?feature_id=${result.featureId}&sae_id=${vlaSaeId}&llm=${currentLLM}`
    )
      .then((res) => res.json())
      .then((resp_data) => {
        const detail = resp_data.data as FeatureDetailResponse;
        dispatch(setSelectedFeature({ data: detail }));
      });
  };

  // Keyboard shortcut for Ctrl+F and click-outside to close autocomplete
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "f") {
        event.preventDefault();
        setIsSearchPanelOpen(true);
        searchInputRef.current?.focus();
      }
      // Escape to close search panel and autocomplete
      if (event.key === "Escape") {
        if (showAutocomplete) {
          setShowAutocomplete(false);
        } else if (isSearchPanelOpen) {
          setIsSearchPanelOpen(false);
        }
      }
    };

    const handleClickOutside = (event: MouseEvent) => {
      if (
        autocompleteRef.current &&
        !autocompleteRef.current.contains(event.target as Node) &&
        searchInputRef.current &&
        !searchInputRef.current.contains(event.target as Node)
      ) {
        setShowAutocomplete(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isSearchPanelOpen, showAutocomplete]);

  const debouncedUpdateVisibleFeatures = useCallback(
    debounce((featureIds: string[]) => {
      if (
        JSON.stringify(featureIds.sort()) !==
        JSON.stringify(visiblePanelFeatures.sort())
      ) {
        setVisiblePanelFeatures(featureIds);

        dataRef.current = dataRef.current.map((point) => ({
          ...point,
          isVisibleInPanel: featureIds.includes(point.featureId.toString()),
        }));

        handleUpdateHexbins(transformRef.current.k);
      }
    }, 150),
    [visiblePanelFeatures]
  );

  const handleVisibleFeaturesChange = (featureIds: string[]) => {
    debouncedUpdateVisibleFeatures(featureIds);
  };

  React.useEffect(() => {
    if (vlaSaeId) {
      resetAllStates();
      // Now supports all_layers natively - backend combines data from all 18 layers
      fetchScatterData(vlaSaeId, query, currentLLM);
    }
  }, [vlaSaeId, query, currentLLM, conceptMethod, pi05Pathway]);

  React.useEffect(() => {
    if (dataRef.current.length > 0) {
      if (!svgRef.current?.querySelector("g")) {
        if (!svgRef.current) return;
        initD3Chart(
          svgRef,
          dataRef,
          xScale,
          yScale,
          tooltipRef,
          transformRef,
          zoomRef,
          handleUpdateHexbins,
          initZoom,
          hierarchicalClusters,
          selectedClusterRef,
          zoomThresholds
        );
      }
    }
  }, [dataRef.current.length]);

  // Effect to highlight features when a concept is selected
  React.useEffect(() => {
    if (dataRef.current.length > 0) {
      const highlightSet = new Set(highlightedFeatureIds);

      // Update the data with highlight state so it persists through redraws
      dataRef.current = dataRef.current.map((point) => ({
        ...point,
        isConceptHighlighted: highlightSet.has(point.featureId),
      }));

      // Redraw the chart with the new highlight state
      if (svgRef.current) {
        handleUpdateHexbins(transformRef.current.k);
      }
    }
  }, [highlightedFeatureIds]);

  // Effect to re-color points when coloringMode changes
  React.useEffect(() => {
    if (dataRef.current.length > 0 && hierarchicalClusters) {
      const currentLevel = selectedClusterRef.current?.level || "10";

      dataRef.current = dataRef.current.map((point) => {
        if (point.isQuery) return point;

        let color: string;
        if (coloringMode === "layer" && point.layer !== undefined) {
          color = LAYER_COLORS[point.layer] || "#6366f1";
        } else if (coloringMode === "concept") {
          const concept = extractPrimaryConcept(point.description || "");
          const conceptType = concept ? CONCEPT_TYPE_MAP[concept] : "unknown";
          color = CONCEPT_TYPE_COLORS[conceptType] || CONCEPT_TYPE_COLORS.unknown;
        } else {
          // cluster mode - use hierarchical cluster colors
          color = hierarchicalClusters[currentLevel]?.colors?.[point.index] || "#6366f1";
        }

        return { ...point, color };
      });

      // Redraw with new colors
      if (svgRef.current) {
        handleUpdateHexbins(transformRef.current.k);
      }
    }
  }, [coloringMode, hierarchicalClusters]);

  const fetchScatterData = async (saeId: string, queryStr?: string, llmModel?: string) => {
    setIsLoading(true);
    setLoadError(null);
    try {
      const url = new URL(
        `${API_BASE_URL}/api/sae/scatter`,
        window.location.origin
      );
      url.searchParams.append("sae_id", saeId);
      if (queryStr) url.searchParams.append("query", queryStr);
      if (llmModel) url.searchParams.append("llm", llmModel);
      url.searchParams.append("method", conceptMethod);
      url.searchParams.append("pathway", pi05Pathway);
      const response = await fetch(url.toString());

      if (!response.ok) {
        throw new Error(`Failed to load data: ${response.status}`);
      }

      const json: SAEScatterResponse = await response.json();
      console.log(json);

      if (!json.data || !json.data.coordinates || json.data.coordinates.length === 0) {
        setLoadError(`No feature data available for ${saeId.replace("-", " → ")}`);
        setIsLoading(false);
        return;
      }

      const data = json.data;

      const similarFeatureIds = new Set(
        data.query?.nearest_features.map((f) => f.feature_id) || []
      );

      const visibleCoordinates = new Set(
        data.query?.nearest_features.map((f) => f.coordinates.join(",")) || []
      );

      const mainPoints = createMainPoints(
        data,
        visibleCoordinates,
        selectedClusterRef,
        saeId,  // Pass SAE ID to extract layer info
        coloringMode  // Pass coloring mode
      ).map((point: ScatterPoint) => {
        const similarFeature = data.query?.nearest_features.find(
          (f) => f.feature_id === point.featureId.toString()
        );

        return {
          ...point,
          isQuerySimilar: similarFeatureIds.has(point.featureId.toString()),
          similarity: similarFeature?.similarity,
        };
      });

      const queryPoint = createQueryPoint(data, selectedClusterRef);

      dataRef.current = [...mainPoints, ...(queryPoint ? [queryPoint] : [])];

      const clusterLevels = Object.keys(data.hierarchical_clusters);
      const defaultLevel = clusterLevels.includes("10")
        ? "10"
        : clusterLevels[0];
      selectedClusterRef.current = {
        level: defaultLevel,
        data: convertClusterData(defaultLevel, data.hierarchical_clusters),
      };

      setClusterLevels(clusterLevels);
      setHierarchicalClusters(data.hierarchical_clusters);

      setQueryInfo(data.query);
      if (data.query) {
        setSimilarFeatures(data.query.nearest_features);
      } else {
        setSimilarFeatures([]);
      }
    } catch (error: any) {
      console.error("Failed to fetch scatter data:", error);
      setLoadError(error.message || "Failed to load feature data");
    } finally {
      setIsLoading(false);
    }
  };

  const zoomToPoint = (p: ScatterPoint, scale: number = 4) => {
    if (
      !svgRef.current ||
      !xScale.current ||
      !yScale.current ||
      !zoomRef.current
    )
      return;
    const svg = d3.select(svgRef.current);
    const width: number = svgRef.current.clientWidth;
    const height: number = svgRef.current.clientHeight;
    const targetX: number = xScale.current(p.x);
    const targetY: number = yScale.current(p.y);

    svg
      .transition()
      .duration(750)
      .call((transition) =>
        zoomRef.current.transform(
          transition as any,
          d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(scale)
            .translate(-targetX, -targetY)
        )
      );

    const g = svg.select("g");
    g.selectAll(".highlight-point").remove();
    g.append("circle")
      .attr("class", "highlight-point")
      .attr("cx", targetX)
      .attr("cy", targetY)
      .attr("r", 6)
      .style("fill", "none")
      .style("stroke", p.color || "#1976d2")
      .style("stroke-width", 2)
      .style("opacity", 0)
      .transition()
      .duration(300)
      .style("opacity", 1);
  };

  const getFeatureDescription = (featureId: string): string => {
    let description = `Feature ${featureId}`;
    if (dataRef.current.length > 0) {
      const indexInScatterData = dataRef.current.findIndex(
        (p) => p.featureId.toString() === featureId
      );
      if (
        indexInScatterData >= 0 &&
        dataRef.current[indexInScatterData].description
      ) {
        description = dataRef.current[indexInScatterData].description;
      }
    }
    return description;
  };

  const handleFeatureClick = (feature: NearestFeature) => {
    setSelectedFeatureId(feature.feature_id);

    fetch(
      `${API_BASE_URL}/api/feature/detail?feature_id=${feature.feature_id}&sae_id=${vlaSaeId}&llm=${currentLLM}`
    )
      .then((res: Response) => res.json())
      .then((resp_data: any) => {
        dispatch(setSelectedFeature({ data: resp_data.data }));
        const point: ScatterPoint | undefined = dataRef.current.find(
          (p) => p.featureId === Number(feature.feature_id)
        );
        if (!point) return;

        let similar_features: NearestFeature[] = [];
        const rawSimilarFeatures = resp_data.data.raw_stats?.similar_features;
        if (rawSimilarFeatures?.feature_ids) {
          for (
            let i = 0;
            i < rawSimilarFeatures.feature_ids.length;
            i++
          ) {
            const featureId = rawSimilarFeatures.feature_ids[i].toString();

            const description = getFeatureDescription(featureId);

            similar_features.push({
              feature_id: featureId,
              similarity: rawSimilarFeatures.values[i],
              description: description,
              coordinates: [0, 0],
            });
          }
        }

        setFeatureDetails(similar_features);
        zoomToPoint(point);
      });
  };

  const handleUpdateHexbins = (scale: number) => {
    updateHexbins(
      svgRef,
      dataRef,
      xScale,
      yScale,
      tooltipRef,
      selectedClusterRef,
      scale,
      handlePointClick
    );
  };

  const handlePointClick = (point: ScatterPoint) => {
    if (point.isQuery) {
      setSimilarFeatures(queryInfo?.nearest_features ?? []);
      setFeatureDetails([]);
      setSelectedPointId(null);
      dispatch(setSelectedFeature(null));

      dataRef.current = dataRef.current.map((p) => ({
        ...p,
        isSelected: false,
        isSelectedSimilar: false,
      }));

      handleUpdateHexbins(transformRef.current.k);
      return;
    }

    fetch(
      `${API_BASE_URL}/api/feature/detail?feature_id=${point.featureId}&sae_id=${vlaSaeId}&llm=${currentLLM}`
    )
      .then((res) => res.json())
      .then((resp_data) => {
        const detail = resp_data.data as FeatureDetailResponse;
        dispatch(setSelectedFeature({ data: detail }));

        let similar_features: NearestFeature[] = [];
        const rawSimilarFeatures = detail.raw_stats?.similar_features;
        if (rawSimilarFeatures?.feature_ids) {
          for (
            let i = 0;
            i < rawSimilarFeatures.feature_ids.length;
            i++
          ) {
            const featureId = rawSimilarFeatures.feature_ids[i].toString();

            const description = getFeatureDescription(featureId);

            similar_features.push({
              feature_id: featureId,
              similarity: rawSimilarFeatures.values[i],
              description: description,
              coordinates: [0, 0],
            });
          }
        }
        setFeatureDetails(similar_features);

        const similarFeatureIds = rawSimilarFeatures?.feature_ids || []; 

        dataRef.current = dataRef.current.map((p) => {
          const isSelected = p.featureId === point.featureId;
          const isSelectedSimilar =
            similarFeatureIds.includes(p.featureId) &&
            p.featureId !== point.featureId;

          return {
            ...p,
            visible: isSelected || p.visible,
            isSelected: isSelected,
            isSelectedSimilar: isSelectedSimilar,
          };
        });
        setSelectedPointId(point.featureId);
        handleUpdateHexbins(transformRef.current.k);
      });

    zoomToPoint(point);
  };

  const { selectedTokens } = useAppSelector((state) => state.feature);
  const { validateFeatureId } = useAppSelector((state) => state.feature);
  const [_, setTokenAnalysisResults] = useState<TokenAnalysisResults | null>(
    null
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const abortControllerRef = React.useRef<AbortController | null>(null);
  const analyzeTokensRequest = async () => {
    if (!selectedTokens?.length || !validateFeatureId || !vlaSaeId) return;

    let controller: AbortController | null = null;
    try {
      setIsAnalyzing(true);
      setAnalysisError(null);

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }

      controller = new AbortController();
      abortControllerRef.current = controller;
      const signal = controller.signal;

      const requestData = {
        feature_id: validateFeatureId,
        sae_id: vlaSaeId,
        llm: currentLLM,
        selected_prompt_tokens: selectedTokens.map((token) => ({
          prompt: token.prompt,
          token_index: token.token_index + 1,
        })),
      };

      const response = await fetch(
        `${API_BASE_URL}/api/feature/tokens-analysis`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestData),
          signal,
        }
      );

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();

      if (signal.aborted) {
        return;
      }

      setTokenAnalysisResults(data.data);

      const unionIds = new Set(data.data.related_features_union || []);
      const promptTokenFeatures: Array<TokenFeatureInfo> =
        data.data.prompt_token_features || [];

      dataRef.current = dataRef.current.map((point: ScatterPoint) => {
        const pointId = point.featureId.toString();
        const isInUnion = unionIds.has(pointId);

        if (isInUnion) {
          const relatedTokens: {
            prompt: string;
            index: number;
            activation: number;
          }[] = [];
          Object.values(promptTokenFeatures).forEach((ptf) => {
            const ptfTyped = ptf as any;
            const features = ptfTyped.features || [];

            const feature = features.find(
              (f: FeatureActivation) => f.feature_id === pointId
            );

            if (feature) {
              relatedTokens.push({
                prompt: ptfTyped.prompt,
                index: ptfTyped.token_index,
                activation: feature.activation,
              });
            }
          });

          return {
            ...point,
            relatedToken: relatedTokens.length > 0 ? relatedTokens : null,
            visible: point.visible,
          };
        }

        return {
          ...point,
          relatedToken: null,
        };
      });

      if (svgRef.current) {
        handleUpdateHexbins(transformRef.current.k);
      }
    } catch (error: any) {
      if (error.name !== "AbortError") {
        console.error("分析token失败:", error);
        setAnalysisError(error.message || "分析过程中发生错误");
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  React.useEffect(() => {
    const analyzeTokens = async () => {
      await analyzeTokensRequest();
    };

    analyzeTokens();

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort(); 
        abortControllerRef.current = null;
      }
    };
  }, [selectedTokens, validateFeatureId, vlaSaeId]);

  const resetAllStates = () => {
    setSelectedFeatureId(null);
    setSimilarFeatures([]);
    setFeatureDetails([]);
    setQueryInfo(null);
    setSelectedPointId(null);
    setVisiblePanelFeatures([]);
    setTokenAnalysisResults(null);
    setIsAnalyzing(false);
    setAnalysisError(null);
    setClusterLevels([]);
    setHierarchicalClusters({});
    dispatch(setValidateFeatureId(null));
    dispatch(setSelectedTokens([]));

    // Clear search state
    setSearchQuery("");
    setCategoryFilter("all");
    setSearchResults([]);
    setIsSearchPanelOpen(false);
    setAutocompleteSuggestions([]);
    setShowAutocomplete(false);
    setSearchMethod("text");

    dataRef.current = [];
    selectedClusterRef.current = {
      level: "10",
      data: {
        clusterCount: 10,
        labels: [] as number[],
        colors: [] as string[],
        centers: [] as [number, number][],
        topics: {} as Record<string, string[]>,
        topicScores: {} as Record<string, number[]>,
        clusterColors: {} as Record<string, string>,
      },
    };

    dispatch(setSelectedFeature(null));

    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  return (
    <Paper className="h-full overflow-hidden relative">
      {isAnalyzing && (
        <div className="absolute top-0 left-0 right-0 z-50 bg-blue-100 text-blue-700 px-4 py-2">
          Analyzing tokens...
        </div>
      )}
      {analysisError && (
        <div className="absolute top-0 left-0 right-0 z-50 bg-red-100 text-red-700 px-4 py-2">
          {analysisError}
        </div>
      )}
      <div className="absolute top-2 left-2 z-10 bg-white/95 rounded shadow-md border border-gray-200 w-72 flex flex-col">
        {/* Header row: title + coloring mode */}
        <div className="px-3 py-1.5">
          <div className="flex items-center gap-2">
            <Typography
              variant="subtitle1"
              fontWeight="bold"
              sx={{ fontSize: "18px" }}
            >
              Feature Explorer
            </Typography>
            <select
              value={coloringMode}
              onChange={(e) => setColoringMode(e.target.value as "cluster" | "concept" | "layer")}
              className="text-xs px-1 py-0.5 border border-gray-300 rounded bg-white"
            >
              <option value="cluster">Cluster Colors</option>
              <option value="concept">Concept Colors</option>
              {vlaSaeId?.includes('all_layers') && <option value="layer">Layer Colors</option>}
            </select>
          </div>
          {/* Method and Pathway selectors */}
          <div className="flex items-center gap-2 mt-1">
            <Tooltip title="Concept identification method" arrow placement="bottom">
              <select
                value={conceptMethod}
                onChange={(e) => setConceptMethod(e.target.value as ConceptMethod)}
                className="text-xs px-1 py-0.5 border border-gray-300 rounded bg-white"
              >
                <option value="contrastive">Contrastive</option>
                {currentModel === 'pi05' && (
                  <option value="ffn">FFN Projection</option>
                )}
              </select>
            </Tooltip>
            {currentModel === 'pi05' && conceptMethod === 'contrastive' && (
              <Tooltip title="Pi0.5 network pathway" arrow placement="bottom">
                <select
                  value={pi05Pathway}
                  onChange={(e) => setPi05Pathway(e.target.value as Pi05Pathway)}
                  className="text-xs px-1 py-0.5 border border-gray-300 rounded bg-white"
                >
                  <option value="expert">Expert (Action Head)</option>
                  <option value="paligemma">PaliGemma (VLM)</option>
                </select>
              </Tooltip>
            )}
          </div>
          {selectedConcept && (
            <Box display="flex" alignItems="center" gap={1} mt={0.5}>
              <Chip
                label={`${selectedConcept} (${selectedType})`}
                size="small"
                color={selectedType === 'motion' ? 'primary' : selectedType === 'object' ? 'secondary' : 'success'}
                onDelete={() => dispatch(clearSelectedConcept())}
                sx={{ height: 22, fontSize: '11px' }}
              />
              <Typography variant="caption" color="text.secondary">
                {highlightedFeatureIds.length} features
              </Typography>
            </Box>
          )}
        </div>

        {/* Search and Filter Section */}
        <div className="px-3 py-1.5 border-t border-gray-200">
          <div className="relative mb-1.5">
            <div className="flex items-center gap-2">
              <TextField
                inputRef={searchInputRef}
                size="small"
                placeholder="Semantic search... (Ctrl+F)"
                value={searchQuery}
                onChange={handleSearchChange}
                onFocus={() => { setIsSearchPanelOpen(true); setShowAutocomplete(autocompleteSuggestions.length > 0); }}
                sx={{
                  flex: 1,
                  '& .MuiInputBase-root': { height: 30, fontSize: '12px' },
                  '& .MuiInputBase-input': { padding: '4px 8px' },
                }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon sx={{ fontSize: 16, color: isSemanticSearching ? '#3b82f6' : 'gray' }} />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      {isSemanticSearching && (
                        <div className="animate-spin rounded-full h-3 w-3 border-b border-blue-600 mr-1" />
                      )}
                      {searchQuery && (
                        <IconButton size="small" onClick={handleClearSearch} sx={{ padding: '2px' }}>
                          <ClearIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      )}
                    </InputAdornment>
                  ),
                }}
              />
            </div>

            {/* Autocomplete Dropdown */}
            {showAutocomplete && autocompleteSuggestions.length > 0 && (
              <div
                ref={autocompleteRef}
                className="absolute left-0 right-0 z-50 mt-0.5 rounded shadow-lg border border-gray-600 overflow-hidden"
                style={{ backgroundColor: '#1e293b' }}
              >
                <div className="px-2 py-1 border-b border-gray-600">
                  <span className="text-[10px] text-gray-400 uppercase tracking-wide">Suggestions</span>
                </div>
                {autocompleteSuggestions.map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAutocompleteSuggestionClick(suggestion)}
                    className="w-full text-left px-3 py-1.5 text-sm text-white hover:bg-gray-700 transition-colors flex items-center gap-2"
                  >
                    <SearchIcon sx={{ fontSize: 12, color: '#94a3b8' }} />
                    <span className="text-[12px]">{suggestion}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={categoryFilter}
                onChange={handleCategoryChange}
                sx={{
                  height: 26,
                  fontSize: '11px',
                  '& .MuiSelect-select': { padding: '3px 8px' }
                }}
              >
                {CONCEPT_CATEGORIES.map((cat) => (
                  <MenuItem key={cat.value} value={cat.value} sx={{ fontSize: '11px' }}>
                    {cat.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {(searchQuery || categoryFilter !== "all") && (
              <button
                onClick={handleClearSearch}
                className="text-xs text-blue-600 hover:text-blue-800 hover:underline whitespace-nowrap"
              >
                Clear all
              </button>
            )}
          </div>
        </div>

        {/* Search Results */}
        {isSearchPanelOpen && (searchQuery || categoryFilter !== "all") && (
          <div className="max-h-[300px] overflow-y-auto border-t border-gray-200">
            {/* Results count and search method badge */}
            <div className="px-2 py-1 bg-gray-50 border-b border-gray-200 sticky top-0 flex items-center justify-between">
              <Typography variant="caption" color="text.secondary">
                {searchResults.length} feature{searchResults.length !== 1 ? 's' : ''} found
                {searchResults.length === 500 && ' (showing first 500)'}
              </Typography>
              {searchQuery && (
                <Chip
                  label={searchMethod === "semantic" ? "Semantic" : "Text"}
                  size="small"
                  sx={{
                    height: 16,
                    fontSize: '9px',
                    fontWeight: 'bold',
                    backgroundColor: searchMethod === "semantic" ? '#7c3aed' : '#6b7280',
                    color: 'white',
                  }}
                />
              )}
            </div>

            {/* Results list */}
            {searchResults.length > 0 ? (
              <div className="p-1">
                {searchResults.map((result) => (
                  <button
                    key={result.featureId}
                    onClick={() => handleSearchResultClick(result)}
                    className={`w-full text-left p-2 rounded hover:bg-blue-50 transition-colors ${
                      selectedFeatureId === result.featureId.toString() ? 'bg-blue-100' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1.5">
                        <Typography variant="caption" fontWeight="bold" className="text-blue-600">
                          ID: {result.featureId}
                        </Typography>
                        {result.similarity !== undefined && (
                          <span
                            className="text-[9px] px-1 py-0.5 rounded font-mono"
                            style={{
                              backgroundColor: `rgba(124, 58, 237, ${Math.min(result.similarity, 1) * 0.3 + 0.1})`,
                              color: '#6d28d9',
                            }}
                          >
                            {(result.similarity * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <Chip
                        label={result.conceptType}
                        size="small"
                        sx={{
                          height: 18,
                          fontSize: '10px',
                          backgroundColor: CONCEPT_TYPE_COLORS[result.conceptType] || CONCEPT_TYPE_COLORS.unknown,
                          color: 'white',
                        }}
                      />
                    </div>
                    <Typography
                      variant="caption"
                      className="text-gray-600 line-clamp-2"
                      sx={{
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        fontSize: '11px',
                        lineHeight: 1.3,
                      }}
                    >
                      {result.description || 'No description'}
                    </Typography>
                  </button>
                ))}
              </div>
            ) : (
              <div className="p-4 text-center">
                <Typography variant="caption" color="text.secondary">
                  {isSemanticSearching ? 'Searching...' : 'No matching features found'}
                </Typography>
              </div>
            )}
          </div>
        )}

        {/* Collapsed state hint */}
        {!isSearchPanelOpen && !searchQuery && categoryFilter === "all" && (
          <div className="py-1 text-center">
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '10px' }}>
              Ctrl+F to search
            </Typography>
          </div>
        )}
      </div>
      <SimilarFeaturesPanel
        similarFeatures={similarFeatures}
        selectedFeatureId={selectedFeatureId}
        hasQueryPoint={dataRef.current.some((p) => p.isQuery)}
        onFeatureClick={handleFeatureClick}
        onVisibleFeaturesChange={handleVisibleFeaturesChange}
      />
      {/* <FeatureDetailPanel
        similarFeatures={featureDetails}
        selectedFeatureId={selectedFeatureId}
        hasQueryPoint={dataRef.current.some((p) => p.isQuery)}
        onFeatureClick={handleFeatureClick}
        onVisibleFeaturesChange={handleVisibleFeaturesChange}
      /> */}

      <div className="h-full relative">
        {/* Loading State */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-20">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <Typography variant="body2" color="text.secondary">
                Loading features...
              </Typography>
            </div>
          </div>
        )}

        {/* Error State */}
        {loadError && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50 z-20">
            <div className="text-center p-4">
              <Typography variant="body1" color="error" gutterBottom>
                {loadError}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                This layer may not have SAE data available.
              </Typography>
            </div>
          </div>
        )}

        {/* Legend - dynamic based on coloringMode */}
        {!isLoading && !loadError && dataRef.current.length > 0 && (
          <div className="absolute top-2 right-2 z-10 bg-white/90 rounded shadow-md px-2 py-1 flex flex-col items-start border border-gray-200 max-h-[300px] overflow-y-auto">
            <Typography variant="subtitle2" className="mb-1">Legend</Typography>

            {/* Layer Colors - show when in layer coloring mode */}
            {coloringMode === "layer" && vlaSaeId?.includes('all_layers') && (
              <div className="mb-2">
                <Typography variant="caption" className="font-medium text-gray-600">Layers:</Typography>
                <div className="grid grid-cols-3 gap-x-2 gap-y-0.5 mt-1">
                  {Object.entries(LAYER_COLORS).map(([layer, color]) => (
                    <div key={layer} className="flex items-center text-[9px]">
                      <div className="w-2 h-2 rounded-sm mr-1" style={{ backgroundColor: color }} />
                      <span>{layer}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Concept Type Colors - show when in concept coloring mode */}
            {coloringMode === "concept" && (
              <div className="mb-2">
                <Typography variant="caption" className="font-medium text-gray-600">Concept Types:</Typography>
                <div className="flex flex-col gap-0.5 mt-1">
                  {Object.entries(CONCEPT_TYPE_COLORS).map(([type, color]) => (
                    <div key={type} className="flex items-center text-[10px]">
                      <div className="w-2.5 h-2.5 rounded-sm mr-1.5" style={{ backgroundColor: color }} />
                      <span className="capitalize">{type}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Cluster coloring info */}
            {coloringMode === "cluster" && (
              <div className="mt-1 text-[10px] text-gray-500">
                Colors = Semantic clusters<br/>
                (SBERT embeddings + hierarchical)
              </div>
            )}

            {/* Concept highlighting legend - show when concept selected */}
            {selectedConcept && (
              <div className="mt-1 pt-1 border-t border-gray-200">
                <div className="flex items-center text-xs">
                  <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: '#ef4444', border: '2px solid #dc2626' }} />
                  <span>Highlighted: {selectedConcept}</span>
                </div>
              </div>
            )}

            {/* Search highlighting legend - show when search is active */}
            {searchResults.length > 0 && (
              <div className="mt-1 pt-1 border-t border-gray-200">
                <div className="flex items-center text-xs">
                  <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: '#f97316', border: '2px solid #ea580c' }} />
                  <span>Search matches: {searchResults.length}</span>
                </div>
              </div>
            )}
          </div>
        )}

        <svg ref={svgRef} className="w-full h-full" />
      </div>
    </Paper>
  );
}
