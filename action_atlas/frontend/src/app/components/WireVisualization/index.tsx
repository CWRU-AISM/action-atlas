"use client";
import React, { useEffect, useRef, useState, useCallback } from "react";
import { Paper, Typography, Box, CircularProgress, Tooltip, Chip, Select, MenuItem, FormControl, InputLabel } from "@mui/material";
import * as d3 from "d3";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import { setSelectedFeature, setFeatureLoading } from "@/redux/features/featureSlice";
import { API_BASE_URL } from "@/config/api";

// Layer information — extended with real data fields
interface LayerInfo {
  id: string;
  name: string;
  layer: number;
  type: string;
  pathway?: string; // "vlm" | "expert" | "dit" | "eagle" | "vlsa" | "paligemma" | "action_expert"
  pathwayIndex?: number; // index within the pathway
  feature_count?: number;
  explained_variance?: number;
  l0_sparsity?: number;
  active_features?: number;
  // Real data fields from /api/vla/layer_connections
  r2?: number;
  success_auc?: number;
  motion_features?: number;
  object_features?: number;
  spatial_features?: number;
  dominant_type?: string;
  top_concepts?: ConceptDetail[];
}

interface ConceptDetail {
  concept?: string;
  type: string;
  name: string;
  n_significant?: number;
  count?: number;
  max_score?: number;
  max_cohens_d?: number;
}

// Feature data from layer_features API
interface LayerFeature {
  feature_id: string;
  index: number;
  activation: number;
  description?: string;
}

interface LayerFeatureData {
  layer_id: string;
  total_features: number;
  active_features: number;
  top_features: LayerFeature[];
  explained_variance?: number;
}

// Connection between layers
interface LayerConnection {
  sourceLayer: number;
  targetLayer: number;
  strength: number; // 0-1 based on R2 values
  delta_r2?: number;
  connectionType?: string; // "sequential" | "skip" | "cross_pathway"
}

// Concept colors organized by type
const CONCEPT_COLORS: Record<string, string> = {
  motion: "#ef4444",      // red
  object: "#3b82f6",      // blue
  spatial: "#10b981",     // emerald
  action_phase: "#f59e0b", // amber
  none: "#64748b",        // slate gray
};

const LAYER_TYPE_COLORS: Record<string, string> = {
  action_expert: "#3b82f6",
  vision_encoder: "#10b981",
  text_encoder: "#8b5cf6",
  cross_attention: "#f59e0b",
  llama_layer: "#8b5cf6",      // OpenVLA uses LLaMA
  transformer: "#3b82f6",      // X-VLA TransformerBlocks
  paligemma: "#10b981",        // Pi0.5 vision encoder, SmolVLA VLM
  dit: "#f97316",              // GR00T DiT layers
  eagle: "#8b5cf6",            // GR00T Eagle LM layers
  vlsa: "#06b6d4",             // GR00T VL-SA layers
  default: "#64748b",
};

// Pathway colors for multi-pathway models
const PATHWAY_COLORS: Record<string, string> = {
  paligemma: "#10b981",   // green for vision/VLM
  action_expert: "#3b82f6", // blue for action expert
  vlm: "#10b981",         // green for VLM layers
  expert: "#f59e0b",      // amber for expert layers
  dit: "#f97316",         // orange for DiT
  eagle: "#8b5cf6",       // purple for Eagle LM
  vlsa: "#06b6d4",        // cyan for VL-SA
};

// Model architecture configurations
const MODEL_ARCHITECTURES = {
  pi05: {
    name: "Pi0.5",
    backbone: "PaliGemma + Action Expert",
    layers: 18,
    layerPrefix: "action_expert_layer",
    layerType: "action_expert",
    hiddenDim: 1024,
    saeWidth: 16384,
    suites: ["expert_pathway"],
    layout: "dual_sequential" as const,
    components: [
      { name: "PaliGemma Vision", type: "vision_encoder", layers: "Frozen" },
      { name: "Action Expert", type: "action_expert", layers: "18 layers" },
      { name: "Action Head", type: "cross_attention", layers: "7-DoF output" },
    ],
    pathways: [
      { name: "PaliGemma VLM", key: "paligemma", count: 18, color: "#10b981", shape: "circle" as const },
      { name: "Gemma Expert", key: "action_expert", count: 18, color: "#3b82f6", shape: "rounded_rect" as const },
    ],
  },
  openvla: {
    name: "OpenVLA-OFT",
    backbone: "LLaMA-7B",
    layers: 32,
    layerPrefix: "layer",
    layerType: "llama_layer",
    hiddenDim: 4096,
    saeWidth: 4096,
    suites: ["libero_goal", "libero_spatial", "libero_object", "libero_10"],
    layout: "single_deep" as const,
    components: [
      { name: "Vision Encoder", type: "vision_encoder", layers: "DinoV2+SigLIP" },
      { name: "LLaMA Backbone", type: "llama_layer", layers: "32 layers" },
      { name: "MLP Action Head", type: "cross_attention", layers: "L1 regression" },
    ],
    pathways: [
      { name: "LLaMA-2", key: "llama", count: 32, color: "#8b5cf6", shape: "tall_circle" as const },
    ],
  },
  xvla: {
    name: "X-VLA",
    backbone: "Florence-2 + TransformerBlocks",
    layers: 24,
    layerPrefix: "layer",
    layerType: "transformer",
    hiddenDim: 1024,
    saeWidth: 8192,
    suites: ["libero_goal", "libero_spatial", "libero_object", "libero_10", "simplerenv_widowx", "simplerenv_google_robot"],
    layout: "single_flow" as const,
    components: [
      { name: "Florence-2 Vision", type: "vision_encoder", layers: "Frozen" },
      { name: "Soft-Prompted Transformer", type: "transformer", layers: "24 layers" },
      { name: "Flow-Matching Head", type: "cross_attention", layers: "7-DoF output" },
    ],
    pathways: [
      { name: "Florence-2", key: "transformer", count: 24, color: "#3b82f6", shape: "rounded_rect" as const },
    ],
  },
  smolvla: {
    name: "SmolVLA",
    backbone: "Interleaved VLM + Expert",
    layers: 64,
    layerPrefix: "vlm_layer",
    layerType: "transformer",
    hiddenDim: 960,
    saeWidth: 7680,
    suites: ["libero_goal", "libero_spatial", "libero_object", "libero_10", "metaworld"],
    layout: "interleaved" as const,
    components: [
      { name: "SmolVLM Vision", type: "vision_encoder", layers: "Frozen" },
      { name: "VLM Pathway", type: "paligemma", layers: "32 layers (960-dim)" },
      { name: "Expert Pathway", type: "action_expert", layers: "32 layers (480-dim)" },
      { name: "Action Head", type: "cross_attention", layers: "Continuous output" },
    ],
    pathways: [
      { name: "VLM", key: "vlm", count: 32, color: "#10b981", shape: "rounded_rect" as const },
      { name: "Expert", key: "expert", count: 32, color: "#f59e0b", shape: "diamond" as const },
    ],
  },
  groot: {
    name: "GR00T N1.5",
    backbone: "DiT + Eagle LM + VL-SA",
    layers: 32,
    layerPrefix: "dit_layer",
    layerType: "transformer",
    hiddenDim: 2048,
    saeWidth: 16384,
    suites: ["libero_object", "libero_goal", "libero_long"],
    layout: "triple" as const,
    components: [
      { name: "Vision Encoder", type: "vision_encoder", layers: "Multi-view" },
      { name: "DiT Blocks", type: "dit", layers: "16 layers" },
      { name: "Eagle LM", type: "eagle", layers: "12 layers" },
      { name: "VL-SA", type: "vlsa", layers: "4 layers" },
      { name: "Action Head", type: "action_expert", layers: "Diffusion" },
    ],
    pathways: [
      { name: "DiT", key: "dit", count: 16, color: "#f97316", shape: "hexagon" as const },
      { name: "Eagle LM", key: "eagle", count: 12, color: "#8b5cf6", shape: "circle" as const },
      { name: "VL-SA", key: "vlsa", count: 4, color: "#06b6d4", shape: "square" as const },
    ],
  },
  act: {
    name: "ACT-ALOHA",
    backbone: "CVAE + Transformer",
    layers: 0,
    layerPrefix: "encoder_layer",
    layerType: "encoder",
    hiddenDim: 512,
    saeWidth: 0,
    suites: ["aloha_screw", "aloha_thread"],
    layout: "encoder_decoder" as const,
    components: [
      { name: "ResNet-18 Vision", type: "vision_encoder", layers: "Frozen" },
      { name: "CVAE Encoder", type: "encoder", layers: "4 layers" },
      { name: "Transformer Decoder", type: "cross_attention", layers: "Action Chunking" },
    ],
    pathways: [] as { name: string; key: string; count: number; color: string; shape: string }[],
  },
} as const;

type NodeShape = "circle" | "rounded_rect" | "diamond" | "hexagon" | "square" | "tall_circle";

// Helper: draw a node shape at (cx, cy) with given size and color
function drawNodeShape(
  parent: d3.Selection<SVGGElement, unknown, null, undefined>,
  shape: NodeShape,
  cx: number,
  cy: number,
  size: number,
  fillColor: string,
  strokeColor: string,
  strokeWidth: number = 2,
): d3.Selection<any, unknown, null, undefined> {
  switch (shape) {
    case "circle":
      return parent.append("circle")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("r", size)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);

    case "tall_circle":
      // Vertically elongated ellipse for deep models like OFT
      return parent.append("ellipse")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("rx", size * 0.75)
        .attr("ry", size * 1.2)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);

    case "rounded_rect": {
      const w = size * 1.6;
      const h = size * 1.6;
      return parent.append("rect")
        .attr("x", cx - w / 2)
        .attr("y", cy - h / 2)
        .attr("width", w)
        .attr("height", h)
        .attr("rx", size * 0.4)
        .attr("ry", size * 0.4)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);
    }

    case "diamond": {
      const d = size * 1.3;
      return parent.append("polygon")
        .attr("points", `${cx},${cy - d} ${cx + d},${cy} ${cx},${cy + d} ${cx - d},${cy}`)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);
    }

    case "hexagon": {
      const r = size * 1.1;
      const hexPoints = Array.from({ length: 6 }, (_, k) => {
        const angle = (Math.PI / 3) * k - Math.PI / 6;
        return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
      }).join(" ");
      return parent.append("polygon")
        .attr("points", hexPoints)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);
    }

    case "square": {
      const s = size * 1.5;
      return parent.append("rect")
        .attr("x", cx - s / 2)
        .attr("y", cy - s / 2)
        .attr("width", s)
        .attr("height", s)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);
    }

    default:
      return parent.append("circle")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("r", size)
        .attr("fill", fillColor)
        .attr("stroke", strokeColor)
        .attr("stroke-width", strokeWidth);
  }
}

// Helper: determine node shape for a layer based on model and pathway
function getNodeShape(modelType: string, layer: LayerInfo): NodeShape {
  const arch = MODEL_ARCHITECTURES[modelType as keyof typeof MODEL_ARCHITECTURES];
  if (!arch || !('pathways' in arch)) return "circle";

  const pathways = arch.pathways as readonly { name: string; key: string; count: number; color: string; shape: string }[];

  if (layer.pathway) {
    const pw = pathways.find(p => p.key === layer.pathway);
    if (pw) return pw.shape as NodeShape;
  }

  // Fallback shapes by model
  switch (modelType) {
    case "openvla": return "tall_circle";
    case "smolvla": return layer.type === "paligemma" ? "rounded_rect" : "diamond";
    case "groot": {
      if (layer.type === "dit") return "hexagon";
      if (layer.type === "eagle") return "circle";
      if (layer.type === "vlsa") return "square";
      return "circle";
    }
    default: return "circle";
  }
}

// Helper: get pathway color for a layer
function getPathwayColor(modelType: string, layer: LayerInfo): string {
  if (layer.pathway && PATHWAY_COLORS[layer.pathway]) {
    return PATHWAY_COLORS[layer.pathway];
  }
  return LAYER_TYPE_COLORS[layer.type] || LAYER_TYPE_COLORS.default;
}

export default function WireVisualization() {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Redux state
  const dispatch = useAppDispatch();
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const selectedAttrs = useAppSelector((state) => state.model.selectedAttrs);

  // Local state
  const [layers, setLayers] = useState<LayerInfo[]>([]);
  const [layerFeatures, setLayerFeatures] = useState<Map<string, LayerFeatureData>>(new Map());
  const [connections, setConnections] = useState<LayerConnection[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<LayerInfo | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<LayerInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [viewMode, setViewMode] = useState<"flow" | "matrix">("flow");
  const [tooltipData, setTooltipData] = useState<{ x: number; y: number; layer: LayerInfo } | null>(null);
  const [selectedSuite, setSelectedSuite] = useState("libero_goal");
  const [dataSource, setDataSource] = useState<"real" | "fallback">("fallback");

  // Fetch real layer connection data from backend
  useEffect(() => {
    setLoading(true);
    setDataSource("fallback");

    const arch = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.pi05;
    const suiteParam = currentModel === "pi05" ? "expert_pathway" : selectedSuite;

    fetch(`${API_BASE_URL}/api/vla/layer_connections?model=${currentModel}&suite=${suiteParam}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 200 && data.data) {
          const apiLayers = data.data.layers || [];
          const apiConnections = data.data.connections || [];

          // Transform API layers to LayerInfo format
          // Backend returns motion_features for OFT/Pi0.5, but total_motion.value for X-VLA/SmolVLA/GR00T
          const layerData: LayerInfo[] = apiLayers.map((l: any) => ({
            id: l.id,
            name: l.name || `${arch.name} Layer ${l.layer}`,
            layer: l.layer,
            type: l.type || arch.layerType,
            pathway: l.pathway,
            pathwayIndex: l.pathway_index,
            feature_count: l.feature_count || 0,
            r2: l.r2,
            success_auc: l.success_auc,
            motion_features: l.motion_features ?? l.total_motion?.value,
            object_features: l.object_features ?? l.total_object?.value,
            spatial_features: l.spatial_features ?? l.total_spatial?.value,
            dominant_type: l.dominant_type,
            top_concepts: l.top_concepts || [],
          }));

          // Transform API connections to LayerConnection format
          const connData: LayerConnection[] = apiConnections.map((c: any) => ({
            sourceLayer: c.source,
            targetLayer: c.target,
            strength: c.strength,
            delta_r2: c.delta_r2,
            connectionType: c.type,
          }));

          if (layerData.length > 0) {
            setLayers(layerData);
            setConnections(connData);
            setDataSource("real");
          } else {
            // Fallback to generated data
            loadFallbackData(arch);
          }
        } else {
          const arch2 = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.pi05;
          loadFallbackData(arch2);
        }
      })
      .catch((error) => {
        console.error("Failed to fetch layer connections:", error);
        loadFallbackData(arch);
      })
      .finally(() => setLoading(false));
  }, [currentModel, selectedSuite]);

  // Fallback: generate layers from architecture config with pathway information
  const loadFallbackData = (arch: typeof MODEL_ARCHITECTURES[keyof typeof MODEL_ARCHITECTURES]) => {
    const defaultLayers: LayerInfo[] = [];
    const fallbackConns: LayerConnection[] = [];

    if ('pathways' in arch && arch.pathways.length > 0) {
      // Multi-pathway models
      let globalIdx = 0;
      const pathways = arch.pathways as readonly { name: string; key: string; count: number; color: string; shape: string }[];

      for (const pw of pathways) {
        for (let i = 0; i < pw.count; i++) {
          defaultLayers.push({
            id: `${pw.key}_layer_${i}`,
            name: `${pw.name} Layer ${i}`,
            layer: globalIdx,
            type: pw.key,
            pathway: pw.key,
            pathwayIndex: i,
            feature_count: arch.saeWidth || 0,
            dominant_type: pw.key || arch.layerType,
          });

          // Sequential connection within pathway
          if (i > 0) {
            fallbackConns.push({
              sourceLayer: globalIdx - 1,
              targetLayer: globalIdx,
              strength: 0.5,
              connectionType: "sequential",
            });
          }
          globalIdx++;
        }
      }

      // For interleaved models (SmolVLA), add cross-pathway connections
      if (arch.layout === "interleaved" && pathways.length === 2) {
        const pw0Count = pathways[0].count;
        const pw1Count = pathways[1].count;
        const minCount = Math.min(pw0Count, pw1Count);
        for (let i = 0; i < minCount; i++) {
          fallbackConns.push({
            sourceLayer: i, // VLM layer i
            targetLayer: pw0Count + i, // Expert layer i
            strength: 0.4,
            connectionType: "cross_pathway",
          });
        }
      }
    } else if (arch.layers > 0) {
      // Single-pathway models
      for (let i = 0; i < arch.layers; i++) {
        defaultLayers.push({
          id: `${arch.layerPrefix}_${i}`,
          name: `${arch.name} Layer ${i}`,
          layer: i,
          type: arch.layerType,
          feature_count: arch.saeWidth || 0,
          dominant_type: arch.layerType,
        });
      }

      // Generate fallback connections
      for (let i = 0; i < defaultLayers.length - 1; i++) {
        fallbackConns.push({
          sourceLayer: defaultLayers[i].layer,
          targetLayer: defaultLayers[i + 1].layer,
          strength: 0.5,
          connectionType: "sequential",
        });
      }
    }

    setLayers(defaultLayers);
    setConnections(fallbackConns);
    setDataSource("fallback");
  };

  // Fetch features for a specific layer
  const fetchLayerFeatures = useCallback(async (layerId: string) => {
    if (layerFeatures.has(layerId)) return;

    setLoadingFeatures(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/layer_features/${layerId}?model=${currentModel}&suite=${selectedSuite}`);
      const data = await res.json();

      if (data.status === 200 && data.data) {
        setLayerFeatures(prev => new Map(prev).set(layerId, data.data));
      }
    } catch (error) {
      console.error(`Failed to fetch features for ${layerId}:`, error);
    } finally {
      setLoadingFeatures(false);
    }
  }, [layerFeatures, currentModel, selectedSuite]);

  // Handle layer click
  const handleLayerClick = useCallback((layer: LayerInfo) => {
    setSelectedLayer(prev => prev?.id === layer.id ? null : layer);
    fetchLayerFeatures(layer.id);
  }, [fetchLayerFeatures]);

  // Handle resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: Math.max(containerRef.current.clientHeight, 400),
        });
      }
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // ============================================================
  // MAIN DRAW FUNCTION
  // ============================================================
  useEffect(() => {
    if (!svgRef.current || dimensions.width === 0) return;

    const arch = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.pi05;

    // ---- ACT: Encoder-Decoder architecture diagram ----
    if (currentModel === "act") {
      drawACTArchitecture();
      return;
    }

    if (layers.length === 0) return;

    // Route to model-specific layout
    switch (arch.layout) {
      case "dual_sequential":
        drawDualSequential();
        break;
      case "single_deep":
        drawSingleDeep();
        break;
      case "single_flow":
        drawSingleFlow();
        break;
      case "interleaved":
        drawInterleaved();
        break;
      case "triple":
        drawTriple();
        break;
      default:
        drawGenericLayers();
        break;
    }

  }, [layers, connections, selectedLayer, hoveredLayer, isExpanded, dimensions, currentModel, dataSource, handleLayerClick]);

  // ============================================================
  // Common SVG setup helper
  // ============================================================
  function setupSvg() {
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();

    // Compute minimum dimensions based on model complexity
    const arch = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.pi05;
    let minWidth = dimensions.width;
    let minHeight = Math.max(dimensions.height, 400);

    // For models with many nodes horizontally, enforce a minimum width
    // so nodes don't bunch up
    const nodeSpacing = 40; // minimum pixels between node centers
    if (arch.layout === "interleaved") {
      // SmolVLA: 32 pairs, need space for each
      const pairCount = Math.max(...(arch.pathways as readonly { count: number }[]).map(p => p.count));
      minWidth = Math.max(minWidth, pairCount * nodeSpacing + 180);
      minHeight = Math.max(minHeight, 450);
    } else if (arch.layout === "triple") {
      // GR00T: 3 tracks, max 16 nodes wide
      const maxTrack = Math.max(...(arch.pathways as readonly { count: number }[]).map(p => p.count));
      minWidth = Math.max(minWidth, maxTrack * nodeSpacing + 180);
      minHeight = Math.max(minHeight, 500);
    } else if (arch.layout === "single_flow") {
      // X-VLA: 24 layers
      minWidth = Math.max(minWidth, layers.length * nodeSpacing + 180);
    } else if (arch.layout === "single_deep") {
      // OFT: 32 layers
      minWidth = Math.max(minWidth, layers.length * (nodeSpacing - 5) + 180);
    }

    const width = minWidth;
    const height = minHeight;
    const margin = { top: 50, right: 60, bottom: 60, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // If the computed dimensions exceed the container, scale down via viewBox
    // instead of overflowing. Otherwise use exact pixel dimensions.
    const fitsInContainer = width <= dimensions.width && height <= dimensions.height;
    if (fitsInContainer) {
      svg
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");
    } else {
      // Scale the SVG to fit inside the container while preserving aspect ratio
      const containerW = dimensions.width || width;
      const containerH = Math.max(dimensions.height, 400);
      svg
        .attr("width", containerW)
        .attr("height", containerH)
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");
    }

    // Dark background
    svg.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "#0f172a");

    // CSS animations
    svg.append("style").text(`
      @keyframes flowDash { to { stroke-dashoffset: -12; } }
      @keyframes pulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
      @keyframes flowPulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 0.8; } }
    `);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    return { svg, g, width, height, margin, innerWidth, innerHeight };
  }

  // ============================================================
  // Common: draw title and subtitle
  // ============================================================
  function drawTitle(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    innerWidth: number,
    title: string,
    subtitle: string,
  ) {
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", -25)
      .attr("text-anchor", "middle")
      .attr("font-size", 14)
      .attr("font-weight", "600")
      .attr("fill", "#e2e8f0")
      .text(title);

    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .attr("font-size", 11)
      .attr("fill", "#64748b")
      .text(subtitle);
  }

  // ============================================================
  // Common: attach node interactivity (hover, click, tooltip)
  // ============================================================
  function attachNodeInteraction(
    el: d3.Selection<any, unknown, null, undefined>,
    layer: LayerInfo,
    cx: number,
    cy: number,
    size: number,
  ) {
    el
      .style("cursor", "pointer")
      .on("mouseenter", function(event: MouseEvent) {
        setHoveredLayer(layer);
        setTooltipData({ x: event.pageX, y: event.pageY, layer });
        d3.select(this)
          .transition()
          .duration(150)
          .attr("transform", `translate(${cx},${cy}) scale(1.15)`)
      })
      .on("mousemove", function(event: MouseEvent) {
        setTooltipData({ x: event.pageX, y: event.pageY, layer });
      })
      .on("mouseleave", function() {
        setHoveredLayer(null);
        setTooltipData(null);
        d3.select(this)
          .transition()
          .duration(150)
          .attr("transform", `translate(${cx},${cy}) scale(1)`)
      })
      .on("click", (event: MouseEvent) => {
        event.stopPropagation();
        handleLayerClick(layer);
      });
  }

  // ============================================================
  // Common: draw a node with shape, label, pie chart, glow
  // ============================================================
  function drawLayerNode(
    nodeGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
    layer: LayerInfo,
    cx: number,
    cy: number,
    size: number,
    shape: NodeShape,
    pathwayColor: string,
  ) {
    const isSelected = selectedLayer?.id === layer.id;
    const isHovered = hoveredLayer?.id === layer.id;

    // Color by dominant concept type if available
    const dominantColor = layer.dominant_type && layer.dominant_type !== "none"
      ? CONCEPT_COLORS[layer.dominant_type] || CONCEPT_COLORS.none
      : pathwayColor;

    const strokeColor = isSelected ? "#ef4444" : dominantColor;
    const strokeW = isSelected ? 3 : 2;

    // Outer glow
    if (isSelected || isHovered) {
      drawNodeShape(
        nodeGroup, shape, cx, cy, size + 6,
        "none",
        isSelected ? "#ef4444" : dominantColor,
        2,
      ).attr("stroke-opacity", 0.4)
       .attr("filter", "blur(4px)")
       .style("pointer-events", "none");
    }

    // Wrap in a group for scaling on hover
    const nodeG = nodeGroup.append("g")
      .attr("transform", `translate(${cx},${cy}) scale(1)`);

    // Main shape (drawn at origin since group is translated)
    const mainNode = drawNodeShape(
      nodeG, shape, 0, 0, size,
      "#1e293b", strokeColor, strokeW,
    );

    // Attach hover/click to the group
    attachNodeInteraction(nodeG, layer, cx, cy, size);

    // Concept composition pie chart inside node
    const motionF = layer.motion_features || 0;
    const objectF = layer.object_features || 0;
    const spatialF = layer.spatial_features || 0;
    const totalF = motionF + objectF + spatialF;

    if (totalF > 0 && size > 8) {
      const pieData = [
        { value: motionF, color: CONCEPT_COLORS.motion },
        { value: objectF, color: CONCEPT_COLORS.object },
        { value: spatialF, color: CONCEPT_COLORS.spatial },
      ].filter(d => d.value > 0);

      const pie = d3.pie<{ value: number; color: string }>()
        .value(d => d.value)
        .sort(null);

      const arcGen = d3.arc<d3.PieArcDatum<{ value: number; color: string }>>()
        .innerRadius(size * 0.45)
        .outerRadius(size * 0.8);

      const pieGroup = nodeG.append("g")
        .style("pointer-events", "none");

      pieGroup.selectAll("path")
        .data(pie(pieData))
        .enter()
        .append("path")
        .attr("d", arcGen as any)
        .attr("fill", d => d.data.color)
        .attr("opacity", 0.7);
    }

    // Layer number label
    const labelIdx = layer.pathwayIndex !== undefined ? layer.pathwayIndex : layer.layer;
    nodeG.append("text")
      .attr("x", 0)
      .attr("y", 0)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", Math.min(10, size * 0.8))
      .attr("font-weight", "bold")
      .attr("fill", "#e2e8f0")
      .style("pointer-events", "none")
      .text(labelIdx);

    return nodeG;
  }

  // ============================================================
  // Common: draw animated flow particles along a connection line
  // ============================================================
  function drawFlowParticles(
    flowGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
    sourceX: number, sourceY: number,
    targetX: number, targetY: number,
    strength: number,
    color: string = "#60a5fa",
  ) {
    if (!isExpanded) return;

    // Animated dashed line
    flowGroup.append("line")
      .attr("x1", sourceX)
      .attr("y1", sourceY)
      .attr("x2", targetX)
      .attr("y2", targetY)
      .attr("stroke", color)
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.3)
      .attr("stroke-dasharray", "4,8")
      .style("animation", `flowDash ${Math.max(0.5, 2 - strength * 1.5)}s linear infinite`);

    // Particles
    const speed = Math.max(600, 2000 - strength * 1500);
    for (let p = 0; p < 2; p++) {
      const particle = flowGroup.append("circle")
        .attr("r", 2)
        .attr("fill", color)
        .attr("opacity", 0.8);

      const animate = () => {
        particle
          .attr("cx", sourceX)
          .attr("cy", sourceY)
          .attr("opacity", 0.8)
          .transition()
          .duration(speed)
          .ease(d3.easeLinear)
          .attr("cx", targetX)
          .attr("cy", targetY)
          .attr("opacity", 0.3)
          .on("end", () => {
            requestAnimationFrame(animate);
          });
      };
      setTimeout(animate, p * (speed / 2));
    }
  }

  // ============================================================
  // Common: draw pathway legend
  // ============================================================
  function drawPathwayLegend(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    innerWidth: number,
    innerHeight: number,
    items: { label: string; color: string; shape: NodeShape }[],
  ) {
    const legendGroup = g.append("g")
      .attr("transform", `translate(${innerWidth - items.length * 70}, ${innerHeight + 35})`);

    items.forEach((item, idx) => {
      const xOff = idx * 70;
      drawNodeShape(
        legendGroup as any, item.shape,
        xOff, 0, 5,
        item.color, item.color, 1.5,
      ).attr("opacity", 0.8);

      legendGroup.append("text")
        .attr("x", xOff + 9)
        .attr("y", 4)
        .attr("font-size", 9)
        .attr("fill", "#94a3b8")
        .text(item.label);
    });
  }

  // ============================================================
  // ACT: Encoder-Decoder Architecture
  // ============================================================
  function drawACTArchitecture() {
    const { g, innerWidth, innerHeight } = setupSvg();

    const components = [
      { name: "ResNet-18\nVision", shape: "rounded_rect" as NodeShape, color: "#10b981", x: 0.08, w: 0.12 },
      { name: "Joint\nEncoder", shape: "rounded_rect" as NodeShape, color: "#8b5cf6", x: 0.25, w: 0.10 },
      { name: "CVAE\nEncoder", shape: "diamond" as NodeShape, color: "#f59e0b", x: 0.42, w: 0.12 },
      { name: "z ~ N(u,s)", shape: "circle" as NodeShape, color: "#ef4444", x: 0.58, w: 0.08 },
      { name: "Transformer\nDecoder", shape: "hexagon" as NodeShape, color: "#3b82f6", x: 0.72, w: 0.14 },
      { name: "Action\nChunks", shape: "rounded_rect" as NodeShape, color: "#64748b", x: 0.92, w: 0.10 },
    ];

    const centerY = innerHeight / 2;

    // Draw connections between components
    const connGroup = g.append("g");
    for (let i = 0; i < components.length - 1; i++) {
      const srcX = components[i].x * innerWidth + (components[i].w * innerWidth) / 2;
      const tgtX = components[i + 1].x * innerWidth - (components[i + 1].w * innerWidth) / 2;

      connGroup.append("line")
        .attr("x1", srcX + 10)
        .attr("y1", centerY)
        .attr("x2", tgtX - 10)
        .attr("y2", centerY)
        .attr("stroke", "#475569")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4,8")
        .style("animation", "flowDash 1.5s linear infinite");

      connGroup.append("polygon")
        .attr("points", `${tgtX - 14},${centerY - 5} ${tgtX - 4},${centerY} ${tgtX - 14},${centerY + 5}`)
        .attr("fill", "#475569");
    }

    // Draw each component with its shape
    components.forEach((comp) => {
      const cx = comp.x * innerWidth;
      const halfW = (comp.w * innerWidth) / 2;
      const halfH = 35;

      const compG = g.append("g")
        .attr("transform", `translate(${cx + halfW},${centerY})`);

      if (comp.shape === "rounded_rect") {
        compG.append("rect")
          .attr("x", -halfW)
          .attr("y", -halfH)
          .attr("width", halfW * 2)
          .attr("height", halfH * 2)
          .attr("rx", 8)
          .attr("fill", "#1e293b")
          .attr("stroke", comp.color)
          .attr("stroke-width", 2);
      } else if (comp.shape === "diamond") {
        compG.append("polygon")
          .attr("points", `0,${-halfH - 5} ${halfW + 5},0 0,${halfH + 5} ${-halfW - 5},0`)
          .attr("fill", "#1e293b")
          .attr("stroke", comp.color)
          .attr("stroke-width", 2);
      } else if (comp.shape === "circle") {
        compG.append("circle")
          .attr("r", halfH + 5)
          .attr("fill", "#1e293b")
          .attr("stroke", comp.color)
          .attr("stroke-width", 2)
          .style("animation", "pulse 2s ease-in-out infinite");
      } else if (comp.shape === "hexagon") {
        const r = halfH + 8;
        const hexPoints = Array.from({ length: 6 }, (_, k) => {
          const angle = (Math.PI / 3) * k - Math.PI / 6;
          return `${r * Math.cos(angle)},${r * Math.sin(angle)}`;
        }).join(" ");
        compG.append("polygon")
          .attr("points", hexPoints)
          .attr("fill", "#1e293b")
          .attr("stroke", comp.color)
          .attr("stroke-width", 2);
      }

      // Label
      const lines = comp.name.split("\n");
      lines.forEach((line, li) => {
        compG.append("text")
          .attr("text-anchor", "middle")
          .attr("y", -4 + li * 14 - (lines.length - 1) * 7)
          .attr("font-size", 11)
          .attr("font-weight", "600")
          .attr("fill", "#e2e8f0")
          .text(line);
      });
    });

    drawTitle(g, innerWidth,
      "ACT-ALOHA (CVAE + Transformer) Architecture",
      "No SAE layer analysis -- interpretability via grid ablation and injection"
    );

    // Bottom labels
    const labels = [
      { x: 0.08, text: "RGB Image", color: "#10b981" },
      { x: 0.25, text: "14-dim joints", color: "#8b5cf6" },
      { x: 0.42, text: "KL divergence", color: "#f59e0b" },
      { x: 0.58, text: "Latent space", color: "#ef4444" },
      { x: 0.72, text: "Cross-attention", color: "#3b82f6" },
      { x: 0.92, text: "100-step chunks", color: "#64748b" },
    ];
    labels.forEach((lbl) => {
      const comp = components.find(c => c.x === lbl.x)!;
      g.append("text")
        .attr("x", lbl.x * innerWidth + (comp.w * innerWidth) / 2)
        .attr("y", centerY + 55)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", lbl.color)
        .attr("opacity", 0.7)
        .text(lbl.text);
    });
  }

  // ============================================================
  // Pi0.5: Dual Sequential (paired nodes per layer — same layer processes both)
  // ============================================================
  function drawDualSequential() {
    const { g, innerWidth, innerHeight } = setupSvg();
    const arch = MODEL_ARCHITECTURES.pi05;

    // Separate layers by pathway or use first 18 for PaliGemma (frozen/no SAE), rest for Expert
    // In practice, the API only returns expert layers. We create PaliGemma placeholder track.
    const expertLayers = layers.filter(l => l.pathway === "action_expert" || !l.pathway);
    const numExpert = expertLayers.length || 18;

    // Paired layout: PaliGemma slightly above center, Expert slightly below
    const centerY = innerHeight / 2;
    const pairGap = 40; // vertical distance between paired nodes
    const topY = centerY - pairGap / 2;   // PaliGemma
    const bottomY = centerY + pairGap / 2; // Expert

    const nodeRadius = Math.min(12, innerWidth / (numExpert * 3.5));

    // X scale
    const xPad = Math.max(nodeRadius * 3, 50);
    const xScale = d3.scaleLinear()
      .domain([0, numExpert - 1])
      .range([xPad, innerWidth - xPad - 40]);

    // Feature count scale
    const maxFeatures = Math.max(...expertLayers.map(l => l.feature_count || 1), 1);
    const sizeScale = d3.scaleLinear()
      .domain([0, maxFeatures])
      .range([nodeRadius * 0.6, nodeRadius * 1.4]);

    const nodeGroup = g.append("g").attr("class", "nodes");
    const connGroup = g.append("g").attr("class", "connections");
    const flowGroup = g.append("g").attr("class", "flow");

    // Draw paired nodes: each layer has a PaliGemma circle above and Expert rounded_rect below
    // connected by a short vertical bridge to show they are the SAME layer
    for (let i = 0; i < numExpert; i++) {
      const cx = xScale(i);

      // --- Vertical bridge connecting the pair (drawn first, behind nodes) ---
      connGroup.append("line")
        .attr("x1", cx)
        .attr("y1", topY + nodeRadius + 1)
        .attr("x2", cx)
        .attr("y2", bottomY - nodeRadius - 1)
        .attr("stroke", "#475569")
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.6);

      // --- PaliGemma node (circle, top) ---
      const paliSize = nodeRadius * 0.8;
      drawNodeShape(
        nodeGroup, "circle", cx, topY, paliSize,
        "#0f172a", arch.pathways[0].color, 1.5,
      ).attr("stroke-opacity", 0.5)
       .attr("stroke-dasharray", "3,3"); // dashed to indicate frozen

      nodeGroup.append("text")
        .attr("x", cx)
        .attr("y", topY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", Math.min(8, nodeRadius * 0.7))
        .attr("fill", "#64748b")
        .style("pointer-events", "none")
        .text(i);

      // --- Expert node (rounded_rect, bottom, with real data) ---
      const expertLayer = expertLayers[i];
      if (expertLayer) {
        const size = sizeScale(expertLayer.feature_count || 1);
        drawLayerNode(nodeGroup, expertLayer, cx, bottomY, size, "rounded_rect", arch.pathways[1].color);
      } else {
        drawNodeShape(
          nodeGroup, "rounded_rect", cx, bottomY, nodeRadius * 0.7,
          "#0f172a", arch.pathways[1].color, 1.5,
        ).attr("stroke-opacity", 0.4);
        nodeGroup.append("text")
          .attr("x", cx)
          .attr("y", bottomY)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", Math.min(8, nodeRadius * 0.7))
          .attr("fill", "#475569")
          .style("pointer-events", "none")
          .text(i);
      }

      // --- Shared layer label centered between the pair ---
      nodeGroup.append("text")
        .attr("x", cx)
        .attr("y", centerY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 7)
        .attr("font-weight", "600")
        .attr("fill", "#94a3b8")
        .style("pointer-events", "none")
        .text(`L${i}`);

      // --- Sequential connections to previous pair ---
      if (i > 0) {
        const prevX = xScale(i - 1);

        // PaliGemma sequential (top track)
        connGroup.append("line")
          .attr("x1", prevX)
          .attr("y1", topY)
          .attr("x2", cx)
          .attr("y2", topY)
          .attr("stroke", arch.pathways[0].color)
          .attr("stroke-width", 1)
          .attr("stroke-opacity", 0.3);

        // Expert sequential (bottom track)
        const conn = connections.find(c => c.sourceLayer === i - 1 && c.targetLayer === i);
        const strength = conn?.strength || 0.5;
        const delta = conn?.delta_r2 ?? 0;

        let connColor: string;
        if (delta > 0.005) connColor = "#10b981";
        else if (delta < -0.005) connColor = "#f59e0b";
        else connColor = "#3b82f6";

        connGroup.append("line")
          .attr("x1", prevX)
          .attr("y1", bottomY)
          .attr("x2", cx)
          .attr("y2", bottomY)
          .attr("stroke", connColor)
          .attr("stroke-width", 1.5)
          .attr("stroke-opacity", 0.6);

        // Flow particles on expert track (every 3rd to avoid clutter)
        if (i % 3 === 0) {
          drawFlowParticles(flowGroup, prevX, bottomY, cx, bottomY, strength, connColor);
        }
      }
    }

    // Track type labels at the left
    g.append("text")
      .attr("x", 2)
      .attr("y", topY)
      .attr("text-anchor", "start")
      .attr("dominant-baseline", "middle")
      .attr("font-size", 9)
      .attr("font-weight", "600")
      .attr("fill", arch.pathways[0].color)
      .text("PaliGemma");

    g.append("text")
      .attr("x", 2)
      .attr("y", bottomY)
      .attr("text-anchor", "start")
      .attr("dominant-baseline", "middle")
      .attr("font-size", 9)
      .attr("font-weight", "600")
      .attr("fill", arch.pathways[1].color)
      .text("Expert");

    // Flow matching annotation at the end
    const headX = innerWidth - 15;
    const headG = g.append("g")
      .attr("transform", `translate(${headX},${centerY})`);
    headG.append("rect")
      .attr("x", -18)
      .attr("y", -22)
      .attr("width", 36)
      .attr("height", 44)
      .attr("rx", 6)
      .attr("fill", "#1e293b")
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2);
    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -5)
      .attr("font-size", 8)
      .attr("font-weight", "bold")
      .attr("fill", "#f59e0b")
      .text("Flow");
    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 8)
      .attr("font-size", 7)
      .attr("fill", "#94a3b8")
      .text("50 steps");

    drawTitle(g, innerWidth,
      `Pi0.5 (PaliGemma + Action Expert) - ${numExpert} Shared Layers`,
      "Dual weights: each layer processes both vision (PaliGemma) and action (Expert) tokens"
    );

    // Bottom labels
    g.append("g")
      .attr("transform", `translate(0,${innerHeight + 20})`)
      .selectAll("text")
      .data(expertLayers)
      .enter()
      .append("text")
      .attr("x", (_, i) => xScale(i))
      .attr("text-anchor", "middle")
      .attr("font-size", 8)
      .attr("fill", (d) => selectedLayer?.id === d.id ? "#ef4444" : "#64748b")
      .text((d) => {
        if (d.r2 !== undefined && d.r2 > 0) {
          return `L${d.pathwayIndex ?? d.layer} (.${(d.r2 * 100).toFixed(0).slice(-2)})`;
        }
        return `L${d.pathwayIndex ?? d.layer}`;
      });

    drawPathwayLegend(g, innerWidth, innerHeight, [
      { label: "PaliGemma", color: arch.pathways[0].color, shape: "circle" },
      { label: "Expert", color: arch.pathways[1].color, shape: "rounded_rect" },
      { label: "Motion", color: CONCEPT_COLORS.motion, shape: "circle" },
      { label: "Object", color: CONCEPT_COLORS.object, shape: "circle" },
      { label: "Spatial", color: CONCEPT_COLORS.spatial, shape: "circle" },
    ]);
  }

  // ============================================================
  // OpenVLA-OFT: Single Deep Stack (tall ellipses, action head)
  // ============================================================
  function drawSingleDeep() {
    const { g, innerWidth, innerHeight } = setupSvg();
    const arch = MODEL_ARCHITECTURES.openvla;

    const nodeRadius = Math.min(14, Math.max(8, innerWidth / (layers.length * 3)));
    const centerY = innerHeight / 2;

    const maxFeatures = Math.max(...layers.map(l => l.feature_count || 1), 1);
    const sizeScale = d3.scaleLinear()
      .domain([0, maxFeatures])
      .range([nodeRadius * 0.6, nodeRadius * 1.5]);

    // X scale with space for vision encoders and action head
    const xPad = Math.max(nodeRadius * 3, 45);
    const xScale = d3.scaleLinear()
      .domain([-1, layers.length])
      .range([xPad, innerWidth - xPad]);

    // R2-based Y offset
    const r2Values = layers.map(l => l.r2 ?? 0);
    const minR2 = Math.min(...r2Values);
    const maxR2 = Math.max(...r2Values);
    const r2Range = maxR2 - minR2 || 1;
    const r2YOffset = (r2: number) => -((r2 - minR2) / r2Range - 0.5) * (innerHeight * 0.04);

    const nodeGroup = g.append("g").attr("class", "nodes");
    const connGroup = g.append("g").attr("class", "connections");
    const flowGroup = g.append("g").attr("class", "flow");

    // Draw grid lines
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(layers)
      .enter()
      .append("line")
      .attr("x1", (_, i) => xScale(i))
      .attr("x2", (_, i) => xScale(i))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "#1e293b")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");

    // Draw R2 profile line
    if (dataSource === "real" && r2Values.some(v => v > 0)) {
      const r2Line = d3.line<number>()
        .x((_, i) => xScale(i))
        .y((r2) => centerY + r2YOffset(r2))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(r2Values)
        .attr("d", r2Line)
        .attr("fill", "none")
        .attr("stroke", "#334155")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "2,3")
        .attr("opacity", 0.5);
    }

    // Vision encoder block at start
    const visionX = xScale(-1);
    const visionG = g.append("g")
      .attr("transform", `translate(${visionX},${centerY})`);
    visionG.append("rect")
      .attr("x", -20)
      .attr("y", -25)
      .attr("width", 40)
      .attr("height", 50)
      .attr("rx", 6)
      .attr("fill", "#0f172a")
      .attr("stroke", "#10b981")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "3,3");
    visionG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -5)
      .attr("font-size", 8)
      .attr("fill", "#10b981")
      .text("DINOv2");
    visionG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 8)
      .attr("font-size", 8)
      .attr("fill", "#10b981")
      .text("+SigLIP");

    // Arrow from vision to first layer
    connGroup.append("line")
      .attr("x1", visionX + 22)
      .attr("y1", centerY)
      .attr("x2", xScale(0) - nodeRadius - 4)
      .attr("y2", centerY + r2YOffset(r2Values[0] || 0))
      .attr("stroke", "#10b981")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,4")
      .attr("stroke-opacity", 0.5);

    // Draw connections
    connections.forEach((conn) => {
      const sourceX = xScale(conn.sourceLayer);
      const targetX = xScale(conn.targetLayer);
      const isSkipConnection = conn.connectionType === "skip" || Math.abs(conn.targetLayer - conn.sourceLayer) > 1;
      const sourceR2 = layers[conn.sourceLayer]?.r2 ?? 0;
      const targetR2 = layers[conn.targetLayer]?.r2 ?? 0;
      const sourceY = centerY + r2YOffset(sourceR2);
      const targetY = centerY + r2YOffset(targetR2);

      const delta = conn.delta_r2 ?? 0;
      let connColor: string;
      if (isSkipConnection) connColor = "#ef4444";
      else if (delta > 0.005) connColor = "#10b981";
      else if (delta < -0.005) connColor = "#f59e0b";
      else connColor = "#8b5cf6"; // LLaMA purple

      const displayStrength = Math.max(0.2, Math.min(1.0, (conn.strength - 0.5) * 2));

      if (isSkipConnection) {
        const midX = (sourceX + targetX) / 2;
        const curveOffset = -30 - (conn.targetLayer - conn.sourceLayer) * 5;
        const path = d3.path();
        path.moveTo(sourceX, sourceY);
        path.quadraticCurveTo(midX, (sourceY + targetY) / 2 + curveOffset, targetX, targetY);

        connGroup.append("path")
          .attr("d", path.toString())
          .attr("fill", "none")
          .attr("stroke", connColor)
          .attr("stroke-width", 2 * displayStrength)
          .attr("stroke-opacity", 0.4);
      } else {
        connGroup.append("line")
          .attr("x1", sourceX)
          .attr("y1", sourceY)
          .attr("x2", targetX)
          .attr("y2", targetY)
          .attr("stroke", connColor)
          .attr("stroke-width", 2 * displayStrength)
          .attr("stroke-opacity", 0.6);

        drawFlowParticles(flowGroup, sourceX, sourceY, targetX, targetY, conn.strength, connColor);
      }
    });

    // Draw layer nodes as tall ellipses
    layers.forEach((layer, i) => {
      const x = xScale(i);
      const layerR2 = layer.r2 ?? 0;
      const y = centerY + r2YOffset(layerR2);
      const size = sizeScale(layer.feature_count || 1);

      drawLayerNode(nodeGroup, layer, x, y, size, "tall_circle", arch.pathways[0].color);
    });

    // MLP Action Head at the end
    const headX = xScale(layers.length);
    const headG = g.append("g")
      .attr("transform", `translate(${headX},${centerY})`);

    // Highlighted action head box
    headG.append("rect")
      .attr("x", -22)
      .attr("y", -28)
      .attr("width", 44)
      .attr("height", 56)
      .attr("rx", 4)
      .attr("fill", "#1e293b")
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2.5);

    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -8)
      .attr("font-size", 9)
      .attr("font-weight", "bold")
      .attr("fill", "#f59e0b")
      .text("MLP");
    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 5)
      .attr("font-size", 8)
      .attr("fill", "#f59e0b")
      .text("Action");
    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 17)
      .attr("font-size", 7)
      .attr("fill", "#94a3b8")
      .text("L1 Regr.");

    // Arrow from last layer to action head
    const lastLayerX = xScale(layers.length - 1);
    const lastR2 = layers[layers.length - 1]?.r2 ?? 0;
    connGroup.append("line")
      .attr("x1", lastLayerX + nodeRadius + 2)
      .attr("y1", centerY + r2YOffset(lastR2))
      .attr("x2", headX - 24)
      .attr("y2", centerY)
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.6);

    // Layer labels at bottom
    g.append("g")
      .attr("transform", `translate(0,${innerHeight + 20})`)
      .selectAll("text")
      .data(layers)
      .enter()
      .append("text")
      .attr("x", (_, i) => xScale(i))
      .attr("text-anchor", "middle")
      .attr("font-size", 8)
      .attr("fill", (d) => selectedLayer?.id === d.id ? "#ef4444" : "#64748b")
      .text((d) => {
        if (d.r2 !== undefined && d.r2 > 0) {
          return `L${d.layer} (.${(d.r2 * 100).toFixed(0).slice(-2)})`;
        }
        return `L${d.layer}`;
      });

    drawTitle(g, innerWidth,
      `OpenVLA-OFT (LLaMA-2 7B) - ${layers.length} Layers + MLP Action Head`,
      dataSource === "real"
        ? "Node height = depth emphasis | Color = dominant concept type | Y = R2"
        : "Single deep pathway with L1 regression action head"
    );

    drawPathwayLegend(g, innerWidth, innerHeight, [
      { label: "LLaMA", color: "#8b5cf6", shape: "tall_circle" },
      { label: "Vision", color: "#10b981", shape: "rounded_rect" },
      { label: "Action Head", color: "#f59e0b", shape: "square" },
      { label: "Motion", color: CONCEPT_COLORS.motion, shape: "circle" },
      { label: "Object", color: CONCEPT_COLORS.object, shape: "circle" },
    ]);
  }

  // ============================================================
  // X-VLA: Single Flow (circles with flow matching output)
  // ============================================================
  function drawSingleFlow() {
    const { g, svg, innerWidth, innerHeight } = setupSvg();

    const centerY = innerHeight / 2;
    const nodeRadius = Math.min(16, Math.max(9, innerWidth / (layers.length * 3)));

    const maxFeatures = Math.max(...layers.map(l => l.feature_count || 1), 1);
    const sizeScale = d3.scaleLinear()
      .domain([0, maxFeatures])
      .range([nodeRadius * 0.6, nodeRadius * 1.4]);

    // X scale with space for input/output
    const xPad = Math.max(nodeRadius * 3, 45);
    const xScale = d3.scaleLinear()
      .domain([-1, layers.length])
      .range([xPad, innerWidth - xPad]);

    // R2-based Y offset
    const r2Values = layers.map(l => l.r2 ?? 0);
    const minR2 = Math.min(...r2Values);
    const maxR2 = Math.max(...r2Values);
    const r2Range = maxR2 - minR2 || 1;
    const r2YOffset = (r2: number) => -((r2 - minR2) / r2Range - 0.5) * (innerHeight * 0.04);

    const nodeGroup = g.append("g").attr("class", "nodes");
    const connGroup = g.append("g").attr("class", "connections");
    const flowGroup = g.append("g").attr("class", "flow");

    // Draw grid
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(layers)
      .enter()
      .append("line")
      .attr("x1", (_, i) => xScale(i))
      .attr("x2", (_, i) => xScale(i))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "#1e293b")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");

    // Florence-2 vision encoder
    const visionX = xScale(-1);
    const visionG = g.append("g")
      .attr("transform", `translate(${visionX},${centerY})`);
    visionG.append("rect")
      .attr("x", -22)
      .attr("y", -22)
      .attr("width", 44)
      .attr("height", 44)
      .attr("rx", 8)
      .attr("fill", "#0f172a")
      .attr("stroke", "#10b981")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "3,3");
    visionG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -3)
      .attr("font-size", 9)
      .attr("fill", "#10b981")
      .text("Florence-2");
    visionG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 10)
      .attr("font-size", 8)
      .attr("fill", "#64748b")
      .text("Vision");

    // Connection from vision
    connGroup.append("line")
      .attr("x1", visionX + 24)
      .attr("y1", centerY)
      .attr("x2", xScale(0) - nodeRadius - 4)
      .attr("y2", centerY + r2YOffset(r2Values[0] || 0))
      .attr("stroke", "#10b981")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,4")
      .attr("stroke-opacity", 0.5);

    // Draw R2 profile
    if (dataSource === "real" && r2Values.some(v => v > 0)) {
      const r2Line = d3.line<number>()
        .x((_, i) => xScale(i))
        .y((r2) => centerY + r2YOffset(r2))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(r2Values)
        .attr("d", r2Line)
        .attr("fill", "none")
        .attr("stroke", "#334155")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "2,3")
        .attr("opacity", 0.5);
    }

    // Draw connections
    connections.forEach((conn) => {
      const sourceX = xScale(conn.sourceLayer);
      const targetX = xScale(conn.targetLayer);
      const sourceR2 = layers[conn.sourceLayer]?.r2 ?? 0;
      const targetR2 = layers[conn.targetLayer]?.r2 ?? 0;
      const sourceY = centerY + r2YOffset(sourceR2);
      const targetY = centerY + r2YOffset(targetR2);

      const delta = conn.delta_r2 ?? 0;
      let connColor: string;
      if (delta > 0.005) connColor = "#10b981";
      else if (delta < -0.005) connColor = "#f59e0b";
      else connColor = "#3b82f6";

      connGroup.append("line")
        .attr("x1", sourceX)
        .attr("y1", sourceY)
        .attr("x2", targetX)
        .attr("y2", targetY)
        .attr("stroke", connColor)
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.6);

      drawFlowParticles(flowGroup, sourceX, sourceY, targetX, targetY, conn.strength, connColor);
    });

    // Draw layer nodes as circles
    layers.forEach((layer, i) => {
      const x = xScale(i);
      const layerR2 = layer.r2 ?? 0;
      const y = centerY + r2YOffset(layerR2);
      const size = sizeScale(layer.feature_count || 1);

      const arch = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.xvla;
      const pathwayShape = (arch.pathways[0]?.shape || "rounded_rect") as NodeShape;
      drawLayerNode(nodeGroup, layer, x, y, size, pathwayShape, arch.pathways[0]?.color || "#3b82f6");
    });

    // Flow matching output with pulsing animation
    const flowEndX = xScale(layers.length);
    const flowG = g.append("g")
      .attr("transform", `translate(${flowEndX},${centerY})`);

    // Pulsing concentric rings for flow matching
    for (let ring = 3; ring >= 1; ring--) {
      flowG.append("circle")
        .attr("r", 12 + ring * 8)
        .attr("fill", "none")
        .attr("stroke", "#f59e0b")
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.15)
        .style("animation", `pulse ${1.5 + ring * 0.5}s ease-in-out infinite`);
    }

    flowG.append("circle")
      .attr("r", 14)
      .attr("fill", "#1e293b")
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2)
      .style("animation", "pulse 2s ease-in-out infinite");

    flowG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -2)
      .attr("font-size", 7)
      .attr("font-weight", "bold")
      .attr("fill", "#f59e0b")
      .text("Flow");
    flowG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 8)
      .attr("font-size", 7)
      .attr("fill", "#f59e0b")
      .text("Match");

    // Arrow from last layer
    const lastLayerX = xScale(layers.length - 1);
    const lastR2 = layers[layers.length - 1]?.r2 ?? 0;
    connGroup.append("line")
      .attr("x1", lastLayerX + nodeRadius + 2)
      .attr("y1", centerY + r2YOffset(lastR2))
      .attr("x2", flowEndX - 16)
      .attr("y2", centerY)
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.6);

    // Layer labels
    g.append("g")
      .attr("transform", `translate(0,${innerHeight + 20})`)
      .selectAll("text")
      .data(layers)
      .enter()
      .append("text")
      .attr("x", (_, i) => xScale(i))
      .attr("text-anchor", "middle")
      .attr("font-size", 8)
      .attr("fill", (d) => selectedLayer?.id === d.id ? "#ef4444" : "#64748b")
      .text((d) => {
        if (d.r2 !== undefined && d.r2 > 0) {
          return `L${d.layer} (.${(d.r2 * 100).toFixed(0).slice(-2)})`;
        }
        return `L${d.layer}`;
      });

    drawTitle(g, innerWidth,
      `X-VLA (Florence-2 + TransformerBlocks) - ${layers.length} Layers`,
      dataSource === "real"
        ? "Soft-prompted Transformer with flow matching output | Y = R2"
        : "Single pathway with flow matching action generation"
    );

    drawPathwayLegend(g, innerWidth, innerHeight, [
      { label: "Florence-2", color: "#10b981", shape: "rounded_rect" },
      { label: "Transformer", color: "#3b82f6", shape: "circle" },
      { label: "Flow Match", color: "#f59e0b", shape: "circle" },
      { label: "Motion", color: CONCEPT_COLORS.motion, shape: "circle" },
      { label: "Object", color: CONCEPT_COLORS.object, shape: "circle" },
    ]);
  }

  // ============================================================
  // SmolVLA: Dual Pathways (independent VLM + Expert tracks with cross-pathway connections)
  // ============================================================
  function drawInterleaved() {
    const { g, innerWidth, innerHeight } = setupSvg();
    const arch = MODEL_ARCHITECTURES.smolvla;

    // Separate VLM and Expert layers
    const vlmLayers = layers.filter(l => l.pathway === "vlm" || l.type === "paligemma" || (l.layer < 32 && !l.pathway));
    const expertLayers = layers.filter(l => l.pathway === "expert" || l.type === "action_expert" || (l.layer >= 32 && !l.pathway));
    const numPairs = Math.max(vlmLayers.length, expertLayers.length, 32);

    const vlmTrackY = innerHeight * 0.30;
    const expertTrackY = innerHeight * 0.70;
    // Smaller nodes to fit 32 per track (nodeRadius ~6-8px)
    const nodeRadius = Math.min(8, Math.max(5, innerWidth / (numPairs * 4.5)));

    const maxFeatures = Math.max(...layers.map(l => l.feature_count || 1), 1);
    const sizeScale = d3.scaleLinear()
      .domain([0, maxFeatures])
      .range([nodeRadius * 0.5, nodeRadius * 1.1]);

    // X scale - use wider padding to avoid edge overlap
    const xPad = Math.max(nodeRadius * 3, 40);
    const xScale = d3.scaleLinear()
      .domain([0, numPairs - 1])
      .range([xPad, innerWidth - xPad - 40]); // extra 40px for action head

    const separatorGroup = g.append("g").attr("class", "separators");
    const nodeGroup = g.append("g").attr("class", "nodes");
    const connGroup = g.append("g").attr("class", "connections");
    const flowGroup = g.append("g").attr("class", "flow");

    // Track labels
    g.append("text")
      .attr("x", 5)
      .attr("y", vlmTrackY - 18)
      .attr("font-size", 10)
      .attr("font-weight", "600")
      .attr("fill", arch.pathways[0].color)
      .text("VLM Pathway (960-dim)");

    g.append("text")
      .attr("x", 5)
      .attr("y", expertTrackY - 18)
      .attr("font-size", 10)
      .attr("font-weight", "600")
      .attr("fill", arch.pathways[1].color)
      .text("Expert Pathway (480-dim)");

    // Draw dashed vertical separators between each pair position
    for (let i = 0; i < numPairs; i++) {
      const cx = xScale(i);
      // Half-way between this position and the next (or at the edge for the last)
      const halfSpacing = i < numPairs - 1
        ? (xScale(i + 1) - cx) / 2
        : (cx - xScale(Math.max(0, i - 1))) / 2;
      const sepX = cx + halfSpacing;

      if (i < numPairs - 1) {
        separatorGroup.append("line")
          .attr("x1", sepX)
          .attr("y1", vlmTrackY - nodeRadius - 8)
          .attr("x2", sepX)
          .attr("y2", expertTrackY + nodeRadius + 8)
          .attr("stroke", "#1e293b")
          .attr("stroke-width", 1)
          .attr("stroke-opacity", 0.5)
          .attr("stroke-dasharray", "2,4");
      }
    }

    // Draw the two independent pathway tracks
    for (let i = 0; i < numPairs; i++) {
      const cx = xScale(i);

      // VLM node (rounded rectangle)
      const vlmLayer = vlmLayers[i];
      if (vlmLayer) {
        const size = sizeScale(vlmLayer.feature_count || 1);
        drawLayerNode(nodeGroup, vlmLayer, cx, vlmTrackY, size, "rounded_rect", arch.pathways[0].color);
      } else {
        // Placeholder
        drawNodeShape(
          nodeGroup, "rounded_rect", cx, vlmTrackY, nodeRadius * 0.6,
          "#0f172a", arch.pathways[0].color, 1,
        ).attr("stroke-opacity", 0.3)
         .attr("stroke-dasharray", "2,2");
        nodeGroup.append("text")
          .attr("x", cx)
          .attr("y", vlmTrackY)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 7)
          .attr("fill", "#475569")
          .text(i);
      }

      // Expert node (diamond)
      const expertLayer = expertLayers[i];
      if (expertLayer) {
        const size = sizeScale(expertLayer.feature_count || 1);
        drawLayerNode(nodeGroup, expertLayer, cx, expertTrackY, size, "diamond", arch.pathways[1].color);
      } else {
        drawNodeShape(
          nodeGroup, "diamond", cx, expertTrackY, nodeRadius * 0.6,
          "#0f172a", arch.pathways[1].color, 1,
        ).attr("stroke-opacity", 0.3)
         .attr("stroke-dasharray", "2,2");
        nodeGroup.append("text")
          .attr("x", cx)
          .attr("y", expertTrackY)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 7)
          .attr("fill", "#475569")
          .text(i);
      }

      // Cross-pathway arrow: VLM[i] -> Expert[i] showing information flow
      // Draw with arrow head to indicate directionality
      const arrowY1 = vlmTrackY + nodeRadius + 3;
      const arrowY2 = expertTrackY - nodeRadius - 3;
      connGroup.append("line")
        .attr("x1", cx)
        .attr("y1", arrowY1)
        .attr("x2", cx)
        .attr("y2", arrowY2)
        .attr("stroke", "#475569")
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.4)
        .attr("stroke-dasharray", "2,4");

      // Small arrow head pointing down
      connGroup.append("polygon")
        .attr("points", `${cx - 2.5},${arrowY2 - 5} ${cx},${arrowY2} ${cx + 2.5},${arrowY2 - 5}`)
        .attr("fill", "#475569")
        .attr("opacity", 0.4);

      // Sequential connections within tracks
      if (i > 0) {
        const prevX = xScale(i - 1);

        // VLM sequential
        connGroup.append("line")
          .attr("x1", prevX)
          .attr("y1", vlmTrackY)
          .attr("x2", cx)
          .attr("y2", vlmTrackY)
          .attr("stroke", arch.pathways[0].color)
          .attr("stroke-width", 1)
          .attr("stroke-opacity", 0.4);

        // Expert sequential
        connGroup.append("line")
          .attr("x1", prevX)
          .attr("y1", expertTrackY)
          .attr("x2", cx)
          .attr("y2", expertTrackY)
          .attr("stroke", arch.pathways[1].color)
          .attr("stroke-width", 1)
          .attr("stroke-opacity", 0.4);
      }
    }

    // Flow particles on tracks (every 4th connection)
    for (let i = 0; i < numPairs - 1; i += 4) {
      drawFlowParticles(flowGroup, xScale(i), vlmTrackY, xScale(i + 1), vlmTrackY, 0.5, arch.pathways[0].color);
      drawFlowParticles(flowGroup, xScale(i), expertTrackY, xScale(i + 1), expertTrackY, 0.5, arch.pathways[1].color);
    }

    // Action head at the end
    const headX = innerWidth - 15;
    const headG = g.append("g")
      .attr("transform", `translate(${headX},${innerHeight / 2})`);
    headG.append("rect")
      .attr("x", -18)
      .attr("y", -22)
      .attr("width", 36)
      .attr("height", 44)
      .attr("rx", 6)
      .attr("fill", "#1e293b")
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2);
    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -5)
      .attr("font-size", 8)
      .attr("font-weight", "bold")
      .attr("fill", "#f59e0b")
      .text("Action");
    headG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 8)
      .attr("font-size", 7)
      .attr("fill", "#94a3b8")
      .text("Cont.");

    drawTitle(g, innerWidth,
      `SmolVLA (VLM + Expert) - 32 Paired Layers`,
      "Dual pathways: VLM and Expert are independent pathways with cross-pathway connections"
    );

    drawPathwayLegend(g, innerWidth, innerHeight, [
      { label: "VLM", color: arch.pathways[0].color, shape: "rounded_rect" },
      { label: "Expert", color: arch.pathways[1].color, shape: "diamond" },
      { label: "Motion", color: CONCEPT_COLORS.motion, shape: "circle" },
      { label: "Object", color: CONCEPT_COLORS.object, shape: "circle" },
      { label: "Spatial", color: CONCEPT_COLORS.spatial, shape: "circle" },
    ]);
  }

  // ============================================================
  // GR00T N1.5: Triple Pathway (proportional track lengths)
  // ============================================================
  function drawTriple() {
    const { g, innerWidth, innerHeight } = setupSvg();
    const arch = MODEL_ARCHITECTURES.groot;

    // Separate layers by pathway
    const ditLayers = layers.filter(l => l.pathway === "dit" || l.type === "dit");
    const eagleLayers = layers.filter(l => l.pathway === "eagle" || l.type === "eagle");
    const vlsaLayers = layers.filter(l => l.pathway === "vlsa" || l.type === "vlsa");

    // If no pathway info, split by index: first 16 = DiT, next 12 = Eagle, last 4 = VL-SA
    const ditCount = ditLayers.length || 16;
    const eagleCount = eagleLayers.length || 12;
    const vlsaCount = vlsaLayers.length || 4;

    // If layers lack pathway, assign them
    let useLayers = layers;
    if (ditLayers.length === 0 && layers.length > 0) {
      useLayers = layers.map((l, i) => {
        if (i < 16) return { ...l, pathway: "dit" as string, pathwayIndex: i };
        else if (i < 28) return { ...l, pathway: "eagle" as string, pathwayIndex: i - 16 };
        else return { ...l, pathway: "vlsa" as string, pathwayIndex: i - 28 };
      });
    }

    const ditLayersResolved = useLayers.filter(l => l.pathway === "dit");
    const eagleLayersResolved = useLayers.filter(l => l.pathway === "eagle");
    const vlsaLayersResolved = useLayers.filter(l => l.pathway === "vlsa");

    // Three tracks with generous vertical spacing
    const trackPadTop = 30;
    const trackPadBottom = 40;
    const usableHeight = innerHeight - trackPadTop - trackPadBottom;
    const trackSpacing = usableHeight / 3;
    const ditTrackY = trackPadTop + trackSpacing * 0.5;
    const eagleTrackY = trackPadTop + trackSpacing * 1.5;
    const vlsaTrackY = trackPadTop + trackSpacing * 2.5;

    const maxTrackLayers = Math.max(ditCount, eagleCount, vlsaCount);
    const nodeRadius = Math.min(14, Math.max(9, innerWidth / (maxTrackLayers * 3.5)));

    const maxFeatures = Math.max(...layers.map(l => l.feature_count || 1), 1);
    const sizeScale = d3.scaleLinear()
      .domain([0, maxFeatures])
      .range([nodeRadius * 0.6, nodeRadius * 1.3]);

    const nodeGroup = g.append("g").attr("class", "nodes");
    const connGroup = g.append("g").attr("class", "connections");
    const flowGroup = g.append("g").attr("class", "flow");

    // Reserve space for action head on the right and labels on the left
    const xPadLeft = Math.max(nodeRadius * 3, 45);
    const xPadRight = 55; // space for action head
    const trackAreaWidth = innerWidth - xPadLeft - xPadRight;

    // Each track gets a proportional X scale based on its own node count
    // All tracks start at the same left edge, but end at proportional distances
    function drawTrack(
      trackLayers: LayerInfo[],
      count: number,
      trackY: number,
      shape: NodeShape,
      color: string,
      label: string,
    ) {
      // Track length is proportional to its node count relative to the max
      const trackWidth = trackAreaWidth * (count / maxTrackLayers);
      const xScaleTrack = d3.scaleLinear()
        .domain([0, count - 1])
        .range([xPadLeft, xPadLeft + trackWidth]);

      // Track label
      g.append("text")
        .attr("x", 5)
        .attr("y", trackY - nodeRadius - 10)
        .attr("font-size", 10)
        .attr("font-weight", "600")
        .attr("fill", color)
        .text(label);

      // Draw nodes
      for (let i = 0; i < count; i++) {
        const cx = xScaleTrack(i);
        const layer = trackLayers[i];

        if (layer) {
          const size = sizeScale(layer.feature_count || 1);
          drawLayerNode(nodeGroup, layer, cx, trackY, size, shape, color);
        } else {
          drawNodeShape(
            nodeGroup, shape, cx, trackY, nodeRadius * 0.7,
            "#0f172a", color, 1.5,
          ).attr("stroke-opacity", 0.4);
          nodeGroup.append("text")
            .attr("x", cx)
            .attr("y", trackY)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", 8)
            .attr("fill", "#475569")
            .text(i);
        }

        // Sequential connections
        if (i > 0) {
          connGroup.append("line")
            .attr("x1", xScaleTrack(i - 1))
            .attr("y1", trackY)
            .attr("x2", cx)
            .attr("y2", trackY)
            .attr("stroke", color)
            .attr("stroke-width", 1.5)
            .attr("stroke-opacity", 0.5);

          if (i % 3 === 0) {
            drawFlowParticles(flowGroup, xScaleTrack(i - 1), trackY, cx, trackY, 0.5, color);
          }
        }
      }

      return xScaleTrack;
    }

    const ditXScale = drawTrack(ditLayersResolved, ditCount, ditTrackY, "hexagon", arch.pathways[0].color, `DiT Blocks (${ditCount} layers)`);
    const eagleXScale = drawTrack(eagleLayersResolved, eagleCount, eagleTrackY, "circle", arch.pathways[1].color, `Eagle LM (${eagleCount} layers)`);
    const vlsaXScale = drawTrack(vlsaLayersResolved, vlsaCount, vlsaTrackY, "square", arch.pathways[2].color, `VL-SA (${vlsaCount} layers)`);

    // Cross-pathway flow arrows: DiT -> Eagle -> VL-SA -> Action Head
    const ditEndX = ditXScale(ditCount - 1);
    const eagleStartX = eagleXScale(0);
    const eagleEndX = eagleXScale(eagleCount - 1);
    const vlsaStartX = vlsaXScale(0);
    const vlsaEndX = vlsaXScale(vlsaCount - 1);

    // DiT -> Eagle: curved arrow from end of DiT track to start of Eagle track
    const connPath1 = d3.path();
    connPath1.moveTo(ditEndX + nodeRadius + 2, ditTrackY);
    connPath1.quadraticCurveTo(
      ditEndX + nodeRadius + 25,
      (ditTrackY + eagleTrackY) / 2,
      eagleStartX, eagleTrackY - nodeRadius - 4
    );
    connGroup.append("path")
      .attr("d", connPath1.toString())
      .attr("fill", "none")
      .attr("stroke", "#64748b")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.5)
      .attr("stroke-dasharray", "4,4");

    // Arrow head: DiT -> Eagle
    connGroup.append("polygon")
      .attr("points", `${eagleStartX - 4},${eagleTrackY - nodeRadius - 10} ${eagleStartX},${eagleTrackY - nodeRadius - 2} ${eagleStartX + 4},${eagleTrackY - nodeRadius - 10}`)
      .attr("fill", "#64748b")
      .attr("opacity", 0.6);

    // Flow label between DiT and Eagle
    const ditEagleMidX = (ditEndX + eagleStartX) / 2 + 15;
    const ditEagleMidY = (ditTrackY + eagleTrackY) / 2;
    connGroup.append("text")
      .attr("x", ditEagleMidX)
      .attr("y", ditEagleMidY - 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 7)
      .attr("fill", "#64748b")
      .attr("opacity", 0.7)
      .text("DiT \u2192 Eagle");

    // Eagle -> VL-SA: curved arrow from end of Eagle track to start of VL-SA track
    const connPath2 = d3.path();
    connPath2.moveTo(eagleEndX + nodeRadius + 2, eagleTrackY);
    connPath2.quadraticCurveTo(
      eagleEndX + nodeRadius + 25,
      (eagleTrackY + vlsaTrackY) / 2,
      vlsaStartX, vlsaTrackY - nodeRadius - 4
    );
    connGroup.append("path")
      .attr("d", connPath2.toString())
      .attr("fill", "none")
      .attr("stroke", "#64748b")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.5)
      .attr("stroke-dasharray", "4,4");

    // Arrow head: Eagle -> VL-SA
    connGroup.append("polygon")
      .attr("points", `${vlsaStartX - 4},${vlsaTrackY - nodeRadius - 10} ${vlsaStartX},${vlsaTrackY - nodeRadius - 2} ${vlsaStartX + 4},${vlsaTrackY - nodeRadius - 10}`)
      .attr("fill", "#64748b")
      .attr("opacity", 0.6);

    // Flow label between Eagle and VL-SA
    const eagleVlsaMidX = (eagleEndX + vlsaStartX) / 2 + 15;
    const eagleVlsaMidY = (eagleTrackY + vlsaTrackY) / 2;
    connGroup.append("text")
      .attr("x", eagleVlsaMidX)
      .attr("y", eagleVlsaMidY - 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 7)
      .attr("fill", "#64748b")
      .attr("opacity", 0.7)
      .text("Eagle \u2192 VL-SA");

    // Diffusion Action Head: all three tracks feed into it
    const diffX = innerWidth - 25;
    const diffG = g.append("g")
      .attr("transform", `translate(${diffX},${innerHeight / 2})`);

    // Pulsing rings for diffusion
    for (let ring = 2; ring >= 1; ring--) {
      diffG.append("circle")
        .attr("r", 10 + ring * 7)
        .attr("fill", "none")
        .attr("stroke", "#f97316")
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.12)
        .style("animation", `pulse ${1.5 + ring * 0.5}s ease-in-out infinite`);
    }
    diffG.append("circle")
      .attr("r", 12)
      .attr("fill", "#1e293b")
      .attr("stroke", "#f97316")
      .attr("stroke-width", 2)
      .style("animation", "pulse 2s ease-in-out infinite");
    diffG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -2)
      .attr("font-size", 7)
      .attr("font-weight", "bold")
      .attr("fill", "#f97316")
      .text("Diff.");
    diffG.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 8)
      .attr("font-size", 7)
      .attr("fill", "#f97316")
      .text("Action");

    // VL-SA -> Action Head (primary output path, solid)
    const vlsaToActionPath = d3.path();
    vlsaToActionPath.moveTo(vlsaEndX + nodeRadius + 2, vlsaTrackY);
    vlsaToActionPath.quadraticCurveTo(
      (vlsaEndX + diffX) / 2,
      (vlsaTrackY + innerHeight / 2) / 2 + 10,
      diffX - 14, innerHeight / 2
    );
    connGroup.append("path")
      .attr("d", vlsaToActionPath.toString())
      .attr("fill", "none")
      .attr("stroke", "#f97316")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.5)
      .attr("stroke-dasharray", "4,4");

    // Arrow head at action head
    connGroup.append("polygon")
      .attr("points", `${diffX - 18},${innerHeight / 2 - 4} ${diffX - 14},${innerHeight / 2} ${diffX - 18},${innerHeight / 2 + 4}`)
      .attr("fill", "#f97316")
      .attr("opacity", 0.6);

    // VL-SA -> Action label
    connGroup.append("text")
      .attr("x", (vlsaEndX + diffX) / 2)
      .attr("y", (vlsaTrackY + innerHeight / 2) / 2 + 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 7)
      .attr("fill", "#f97316")
      .attr("opacity", 0.6)
      .text("VL-SA \u2192 Action");

    drawTitle(g, innerWidth,
      `GR00T N1.5 (DiT + Eagle LM + VL-SA) - 32 Total Layers`,
      "16 DiT diffusion + 12 Eagle language model + 4 VL-SA vision-language attention"
    );

    drawPathwayLegend(g, innerWidth, innerHeight, [
      { label: "DiT", color: arch.pathways[0].color, shape: "hexagon" },
      { label: "Eagle", color: arch.pathways[1].color, shape: "circle" },
      { label: "VL-SA", color: arch.pathways[2].color, shape: "square" },
      { label: "Diffusion", color: "#f97316", shape: "circle" },
      { label: "Motion", color: CONCEPT_COLORS.motion, shape: "circle" },
    ]);
  }

  // ============================================================
  // Generic fallback (same as old behavior)
  // ============================================================
  function drawGenericLayers() {
    const { g, innerWidth, innerHeight } = setupSvg();
    const arch = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.pi05;

    const centerY = innerHeight / 2;
    const nodeRadius = Math.min(18, Math.max(8, innerWidth / (layers.length * 3)));

    const xPad = Math.max(nodeRadius * 2, 40);
    const xScale = d3.scaleLinear()
      .domain([0, layers.length - 1])
      .range([xPad, innerWidth - xPad]);

    const maxFeatures = Math.max(...layers.map(l => l.feature_count || 1), 1);
    const sizeScale = d3.scaleLinear()
      .domain([0, maxFeatures])
      .range([nodeRadius * 0.6, nodeRadius * 1.5]);

    const r2Values = layers.map(l => l.r2 ?? 0);
    const minR2 = Math.min(...r2Values);
    const maxR2 = Math.max(...r2Values);
    const r2Range = maxR2 - minR2 || 1;
    const r2YOffset = (r2: number) => -((r2 - minR2) / r2Range - 0.5) * (innerHeight * 0.03);

    const nodeGroup = g.append("g").attr("class", "nodes");
    const connGroup = g.append("g").attr("class", "connections");
    const flowGroup = g.append("g").attr("class", "flow");

    // Grid
    g.append("g")
      .selectAll("line")
      .data(layers)
      .enter()
      .append("line")
      .attr("x1", (_, i) => xScale(i))
      .attr("x2", (_, i) => xScale(i))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "#1e293b")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");

    // Connections
    connections.forEach((conn) => {
      const sourceX = xScale(conn.sourceLayer);
      const targetX = xScale(conn.targetLayer);
      const sourceR2 = layers[conn.sourceLayer]?.r2 ?? 0;
      const targetR2 = layers[conn.targetLayer]?.r2 ?? 0;
      const sourceY = centerY + r2YOffset(sourceR2);
      const targetY = centerY + r2YOffset(targetR2);

      connGroup.append("line")
        .attr("x1", sourceX)
        .attr("y1", sourceY)
        .attr("x2", targetX)
        .attr("y2", targetY)
        .attr("stroke", "#3b82f6")
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.5);

      drawFlowParticles(flowGroup, sourceX, sourceY, targetX, targetY, conn.strength);
    });

    // Nodes
    layers.forEach((layer, i) => {
      const x = xScale(i);
      const layerR2 = layer.r2 ?? 0;
      const y = centerY + r2YOffset(layerR2);
      const size = sizeScale(layer.feature_count || 1);
      const shape = getNodeShape(currentModel, layer);
      const color = getPathwayColor(currentModel, layer);

      drawLayerNode(nodeGroup, layer, x, y, size, shape, color);
    });

    // Labels
    g.append("g")
      .attr("transform", `translate(0,${innerHeight + 20})`)
      .selectAll("text")
      .data(layers)
      .enter()
      .append("text")
      .attr("x", (_, i) => xScale(i))
      .attr("text-anchor", "middle")
      .attr("font-size", 9)
      .attr("fill", (d) => selectedLayer?.id === d.id ? "#ef4444" : "#64748b")
      .text((d) => {
        if (d.r2 !== undefined && d.r2 > 0) {
          return `L${d.layer} (.${(d.r2 * 100).toFixed(0).slice(-2)})`;
        }
        return `L${d.layer}`;
      });

    drawTitle(g, innerWidth,
      `${arch.name} (${arch.backbone}) - ${layers.length} Layers`,
      isExpanded ? "Click to pause flow" : "Click to resume flow"
    );
  }

  // ============================================================
  // END OF DRAW FUNCTIONS
  // ============================================================

  // Get selected layer features
  const selectedLayerFeatures = selectedLayer ? layerFeatures.get(selectedLayer.id) : null;

  // Available suites for current model
  const arch = MODEL_ARCHITECTURES[currentModel] || MODEL_ARCHITECTURES.pi05;
  const availableSuites = arch.suites || ["libero_goal"];

  return (
    <Paper className="h-full flex flex-col rounded-lg shadow-md bg-[#0f172a]">
      {/* Header */}
      <div className="h-10 flex items-center justify-between px-3 bg-[#0a1628] rounded-t-lg border-b border-slate-700">
        <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
          Layer Circuit Visualization
        </Typography>
        <Box display="flex" gap={1} alignItems="center">
          {loading && <CircularProgress size={16} sx={{ color: '#64748b' }} />}
          {/* Suite selector for models with multiple suites */}
          {availableSuites.length > 1 && (
            <select
              value={selectedSuite}
              onChange={(e) => setSelectedSuite(e.target.value)}
              style={{
                backgroundColor: '#1e293b',
                color: '#94a3b8',
                border: '1px solid #334155',
                borderRadius: 4,
                fontSize: '0.7rem',
                padding: '2px 4px',
                height: 20,
              }}
            >
              {availableSuites.map(s => (
                <option key={s} value={s}>{s.replace('libero_', 'L-').replace('simplerenv_', 'SE-').replace('metaworld_', 'MW-')}</option>
              ))}
            </select>
          )}
          <Chip
            label={`${layers.length} Layers`}
            size="small"
            sx={{
              backgroundColor: '#1e293b',
              color: '#94a3b8',
              fontSize: '0.7rem',
              height: 20,
            }}
          />
          <Chip
            label={dataSource === "real" ? "LIVE DATA" : "NO DATA"}
            size="small"
            sx={{
              backgroundColor: dataSource === "real" ? '#065f46' : '#7f1d1d',
              color: dataSource === "real" ? '#6ee7b7' : '#fca5a5',
              fontSize: '0.6rem',
              height: 18,
            }}
          />
          <Chip
            label={currentModel.toUpperCase()}
            size="small"
            color="error"
            sx={{ fontSize: '0.7rem', height: 20 }}
          />
        </Box>
      </div>

      {/* Main visualization area */}
      <div className="flex flex-1 min-h-0 overflow-auto">
        {/* Left: Circuit visualization */}
        <div
          ref={containerRef}
          className="flex-1 min-w-0 cursor-pointer"
          style={{ minHeight: 400 }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <svg ref={svgRef} width="100%" height="100%" />
        </div>

        {/* Right: Selected layer details */}
        {selectedLayer && (
          <div className="w-64 flex-shrink-0 bg-[#1e293b] border-l border-slate-700 overflow-auto p-3">
            <Typography variant="subtitle2" sx={{ color: '#e2e8f0', mb: 1 }}>
              Layer {selectedLayer.pathwayIndex ?? selectedLayer.layer} Details
              {selectedLayer.pathway && (
                <span style={{ color: PATHWAY_COLORS[selectedLayer.pathway] || '#94a3b8', fontSize: '0.7rem', marginLeft: 6 }}>
                  ({selectedLayer.pathway})
                </span>
              )}
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                ID: {selectedLayer.id}
              </Typography>
            </Box>

            {/* Layer Statistics */}
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, mb: 2 }}>
              <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                  R2 (nsteps)
                </Typography>
                <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 'bold' }}>
                  {selectedLayer.r2 !== undefined ? selectedLayer.r2.toFixed(4) : 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                  Features (d&gt;1)
                </Typography>
                <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 'bold' }}>
                  {selectedLayer.feature_count?.toLocaleString() || 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ color: CONCEPT_COLORS.motion, display: 'block' }}>
                  Motion
                </Typography>
                <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 'bold' }}>
                  {selectedLayer.motion_features ?? 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ color: CONCEPT_COLORS.object, display: 'block' }}>
                  Object
                </Typography>
                <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 'bold' }}>
                  {selectedLayer.object_features ?? 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ color: CONCEPT_COLORS.spatial, display: 'block' }}>
                  Spatial
                </Typography>
                <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 'bold' }}>
                  {selectedLayer.spatial_features ?? 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                  Dominant
                </Typography>
                <Typography variant="body2" sx={{
                  color: selectedLayer.dominant_type ? (CONCEPT_COLORS[selectedLayer.dominant_type] || '#e2e8f0') : '#e2e8f0',
                  fontWeight: 'bold'
                }}>
                  {selectedLayer.dominant_type || 'N/A'}
                </Typography>
              </Box>
              {selectedLayer.success_auc !== undefined && selectedLayer.success_auc > 0 && (
                <Box sx={{ p: 1, bgcolor: '#0f172a', borderRadius: 1, gridColumn: 'span 2' }}>
                  <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                    Success AUC
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 'bold' }}>
                    {selectedLayer.success_auc.toFixed(4)}
                  </Typography>
                </Box>
              )}
            </Box>

            {/* Top Concepts from real data */}
            {selectedLayer.top_concepts && selectedLayer.top_concepts.length > 0 && (
              <>
                <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mb: 1 }}>
                  Top Concepts (by feature count)
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mb: 2 }}>
                  {selectedLayer.top_concepts.slice(0, 6).map((concept, idx) => {
                    const conceptLabel = concept.concept || concept.name || `concept_${idx}`;
                    const featureCount = concept.n_significant ?? concept.count ?? 0;
                    const conceptType = concept.type || 'unknown';
                    const cohensD = concept.max_cohens_d ?? null;
                    const maxScore = concept.max_score ?? null;
                    return (
                      <Box
                        key={conceptLabel}
                        sx={{
                          p: 1,
                          bgcolor: '#0f172a',
                          borderRadius: 1,
                          borderLeft: `3px solid ${CONCEPT_COLORS[conceptType] || '#64748b'}`,
                        }}
                      >
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                            {conceptLabel}
                          </Typography>
                          <Typography variant="caption" sx={{ color: CONCEPT_COLORS[conceptType] || '#60a5fa', fontWeight: 'bold' }}>
                            {featureCount}
                          </Typography>
                        </Box>
                        {(cohensD !== null || maxScore !== null) && (
                          <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.6rem' }}>
                            {cohensD !== null ? `d=${cohensD.toFixed(2)}` : ''}{cohensD !== null && maxScore !== null ? ' | ' : ''}{maxScore !== null ? `score=${maxScore.toFixed(2)}` : ''}
                          </Typography>
                        )}
                      </Box>
                    );
                  })}
                </Box>
              </>
            )}

            {/* Legacy: Top Features from layer_features API */}
            <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mb: 1 }}>
              SAE Features
            </Typography>

            {loadingFeatures ? (
              <Box display="flex" justifyContent="center" p={2}>
                <CircularProgress size={20} sx={{ color: '#64748b' }} />
              </Box>
            ) : selectedLayerFeatures?.top_features ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                {selectedLayerFeatures.top_features.slice(0, 8).map((feature, idx) => (
                  <Box
                    key={feature.feature_id}
                    sx={{
                      p: 1,
                      bgcolor: '#0f172a',
                      borderRadius: 1,
                      cursor: 'pointer',
                      '&:hover': { bgcolor: '#1e3a5f' },
                    }}
                  >
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                        #{idx + 1} F{feature.index}
                      </Typography>
                      <Typography variant="caption" sx={{ color: '#60a5fa', fontWeight: 'bold' }}>
                        {feature.activation.toFixed(3)}
                      </Typography>
                    </Box>
                    {feature.description && (
                      <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.65rem' }}>
                        {feature.description.slice(0, 50)}...
                      </Typography>
                    )}
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                Click to load SAE features
              </Typography>
            )}
          </div>
        )}
      </div>

      {/* Footer - Hover Info */}
      <div className="h-12 flex items-center justify-between px-4 bg-[#0a1628] border-t border-slate-700">
        {hoveredLayer ? (
          <Box display="flex" gap={3} alignItems="center" flexWrap="wrap">
            <Typography variant="caption" sx={{ color: '#64748b' }}>
              Layer <span style={{ color: '#e2e8f0', fontWeight: 'bold' }}>{hoveredLayer.pathwayIndex ?? hoveredLayer.layer}</span>
              {hoveredLayer.pathway && (
                <span style={{ color: PATHWAY_COLORS[hoveredLayer.pathway] || '#94a3b8', marginLeft: 4 }}>
                  ({hoveredLayer.pathway})
                </span>
              )}
            </Typography>
            {hoveredLayer.r2 !== undefined && (
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                R2: <span style={{ color: '#f59e0b', fontWeight: 'bold' }}>{hoveredLayer.r2.toFixed(4)}</span>
              </Typography>
            )}
            {(hoveredLayer.feature_count ?? 0) > 0 && (
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                Features: <span style={{ color: '#10b981' }}>{hoveredLayer.feature_count?.toLocaleString()}</span>
              </Typography>
            )}
            {hoveredLayer.dominant_type && hoveredLayer.dominant_type !== "none" && (
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                Dominant: <span style={{ color: CONCEPT_COLORS[hoveredLayer.dominant_type] || '#e2e8f0' }}>{hoveredLayer.dominant_type}</span>
              </Typography>
            )}
            {(hoveredLayer.motion_features ?? 0) + (hoveredLayer.object_features ?? 0) + (hoveredLayer.spatial_features ?? 0) > 0 && (
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                <span style={{ color: CONCEPT_COLORS.motion }}>M:{hoveredLayer.motion_features}</span>{" "}
                <span style={{ color: CONCEPT_COLORS.object }}>O:{hoveredLayer.object_features}</span>{" "}
                <span style={{ color: CONCEPT_COLORS.spatial }}>S:{hoveredLayer.spatial_features}</span>
              </Typography>
            )}
          </Box>
        ) : (
          <Typography variant="caption" sx={{ color: '#64748b' }}>
            Hover over a layer node for details, click to select
          </Typography>
        )}
        <Box display="flex" gap={2} flexShrink={0}>
          <Box display="flex" alignItems="center" gap={0.5}>
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: CONCEPT_COLORS.motion }} />
            <Typography variant="caption" sx={{ color: '#64748b' }}>Motion</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: CONCEPT_COLORS.object }} />
            <Typography variant="caption" sx={{ color: '#64748b' }}>Object</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: CONCEPT_COLORS.spatial }} />
            <Typography variant="caption" sx={{ color: '#64748b' }}>Spatial</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#ef4444' }} />
            <Typography variant="caption" sx={{ color: '#64748b' }}>Skip</Typography>
          </Box>
        </Box>
      </div>

      {/* Floating tooltip */}
      {tooltipData && (
        <div
          ref={tooltipRef}
          className="fixed z-50 pointer-events-none"
          style={{
            left: tooltipData.x + 10,
            top: tooltipData.y - 10,
            transform: 'translateY(-100%)',
          }}
        >
          <div className="bg-slate-900 border border-slate-700 rounded-lg p-2 shadow-xl">
            <Typography variant="caption" sx={{ color: '#e2e8f0', fontWeight: 'bold', display: 'block' }}>
              Layer {tooltipData.layer.pathwayIndex ?? tooltipData.layer.layer}
              {tooltipData.layer.pathway && (
                <span style={{ color: PATHWAY_COLORS[tooltipData.layer.pathway] || '#94a3b8', marginLeft: 4, fontWeight: 'normal' }}>
                  ({tooltipData.layer.pathway})
                </span>
              )}
            </Typography>
            <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
              {tooltipData.layer.name || tooltipData.layer.id}
            </Typography>
            {tooltipData.layer.r2 !== undefined && (
              <Typography variant="caption" sx={{ color: '#f59e0b', display: 'block' }}>
                R2 = {tooltipData.layer.r2.toFixed(4)}
              </Typography>
            )}
            {(tooltipData.layer.feature_count ?? 0) > 0 && (
              <Typography variant="caption" sx={{ color: '#10b981', display: 'block' }}>
                {tooltipData.layer.feature_count?.toLocaleString()} concept features (d&gt;1)
              </Typography>
            )}
            {tooltipData.layer.dominant_type && tooltipData.layer.dominant_type !== "none" && (
              <Typography variant="caption" sx={{ color: CONCEPT_COLORS[tooltipData.layer.dominant_type] || '#64748b', display: 'block' }}>
                Dominant: {tooltipData.layer.dominant_type}
              </Typography>
            )}
          </div>
        </div>
      )}
    </Paper>
  );
}
