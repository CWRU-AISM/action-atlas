import { styled, TableCell, TableRow } from "@mui/material";
import { ClusterData } from "@/types/sae";
import { ScatterPoint } from "@/types/types";
import React from "react";
import * as d3 from "d3";
import * as d3Hexbin from "d3-hexbin";
import {
  BSplineShapeGenerator,
  BubbleSet,
  PointPath,
  ShapeSimplifier,
} from "bubblesets";

// Layer colors for all_layers mode (18 distinct colors for layers 0-17)
export const LAYER_COLORS: Record<string, string> = {
  "0": "#e6194b",  // Red
  "1": "#3cb44b",  // Green
  "2": "#ffe119",  // Yellow
  "3": "#4363d8",  // Blue
  "4": "#f58231",  // Orange
  "5": "#911eb4",  // Purple
  "6": "#46f0f0",  // Cyan
  "7": "#f032e6",  // Magenta
  "8": "#bcf60c",  // Lime
  "9": "#fabebe",  // Pink
  "10": "#008080", // Teal
  "11": "#e6beff", // Lavender
  "12": "#9a6324", // Brown
  "13": "#fffac8", // Beige
  "14": "#800000", // Maroon
  "15": "#aaffc3", // Mint
  "16": "#808000", // Olive
  "17": "#ffd8b1", // Apricot
};

// Concept type colors for concept coloring mode
export const CONCEPT_TYPE_COLORS: Record<string, string> = {
  motion: "#3b82f6",   // Blue for motion concepts (put, open, push, interact)
  object: "#ef4444",   // Red for object concepts (bowl, wine_bottle, etc.)
  phase: "#10b981",    // Green for action phase concepts (grasp, lift, etc.)
  spatial: "#f59e0b",  // Amber for spatial concepts (on, in, top, etc.)
  unknown: "#9ca3af",  // Gray for unknown
};

// Individual subconcept colors - distinct colors for each concept
export const SUBCONCEPT_COLORS: Record<string, string> = {
  // Motion concepts (blues/purples)
  put: "#3b82f6",      // Blue
  open: "#8b5cf6",     // Violet
  push: "#6366f1",     // Indigo
  interact: "#a855f7", // Purple
  close: "#7c3aed",    // Violet-dark
  // Object concepts (reds/oranges/pinks)
  bowl: "#ef4444",     // Red
  wine_bottle: "#f97316", // Orange
  rack: "#ec4899",     // Pink
  cube: "#f43f5e",     // Rose
  plate: "#fb7185",    // Rose-light
  cabinet: "#e11d48",  // Rose-dark
  door: "#dc2626",     // Red-600
  drawer: "#ea580c",   // Orange-dark
  faucet: "#c2410c",   // Orange-darker
  mug: "#be185d",      // Pink-dark
  stove: "#9f1239",    // Rose-darker
  // Action phase concepts (greens/teals)
  grasp: "#10b981",    // Emerald
  transport: "#14b8a6", // Teal
  lower: "#06b6d4",    // Cyan
  lift: "#22c55e",     // Green
  approach: "#0ea5e9", // Sky
  release: "#059669",  // Emerald-dark
  retract: "#0d9488",  // Teal-dark
  // Spatial concepts (yellows/ambers)
  on: "#f59e0b",       // Amber
  in: "#eab308",       // Yellow
  top: "#fbbf24",      // Amber-light
  under: "#d97706",    // Amber-dark
  above: "#ca8a04",    // Yellow-dark
  below: "#a16207",    // Yellow-darker
  left: "#facc15",     // Yellow-400
  right: "#fde047",    // Yellow-300
  near: "#fef08a",     // Yellow-200
};

// Map known concepts to their types
export const CONCEPT_TYPE_MAP: Record<string, string> = {
  // Motion concepts
  put: "motion", open: "motion", push: "motion", interact: "motion", close: "motion",
  // Object concepts
  bowl: "object", wine_bottle: "object", rack: "object", cube: "object",
  plate: "object", bowl_top: "object", cabinet: "object", door: "object",
  drawer: "object", faucet: "object", mug: "object", stove: "object",
  // Action phase concepts
  grasp: "phase", transport: "phase", lower: "phase", lift: "phase",
  approach: "phase", release: "phase", retract: "phase",
  // Spatial concepts
  on: "spatial", in: "spatial", top: "spatial", under: "spatial",
  above: "spatial", below: "spatial", left: "spatial", right: "spatial", near: "spatial",
};

export const metricOptions = [
  { value: "l0_sparsity", label: "L0 Sparsity" },
  { value: "l2_ratio", label: "L2 Ratio" },
  { value: "explained_variance", label: "Explained Variance" },
  { value: "kl_div_score", label: "KL Divergence" },
  { value: "ce_loss_score", label: "CE Loss" },
  { value: "llm_test_accuracy", label: "LLM Test Acc" },
  { value: "llm_top_1_test_accuracy", label: "LLM Top-1 Acc" },
  { value: "llm_top_2_test_accuracy", label: "LLM Top-2 Acc" },
  { value: "llm_top_5_test_accuracy", label: "LLM Top-5 Acc" },
  { value: "sae_test_accuracy", label: "SAE Test Acc" },
  { value: "sae_top_1_test_accuracy", label: "SAE Top-1 Acc" },
  { value: "sae_top_2_test_accuracy", label: "SAE Top-2 Acc" },
  { value: "sae_top_5_test_accuracy", label: "SAE Top-5 Acc" },
  { value: "scr_metric_threshold_10", label: "SCR (10)" },
  { value: "scr_metric_threshold_20", label: "SCR (20)" },
  { value: "tpp_threshold_10", label: "TPP (10)" },
  { value: "tpp_threshold_20", label: "TPP (20)" },
  { value: "top_10_score", label: "Top 10 Score" },
  { value: "top_100_score", label: "Top 100 Score" },
  { value: "top_1000_score", label: "Top 1000 Score" },
];

export const normalizeValue = (
  value: number | null,
  attr: string,
  modelData: any[]
): number => {
  if (value === null) return 0;
  const values = modelData
    .map((m) => m[attr]?.value)
    .filter((v) => v !== null && !isNaN(v));
  if (values.length === 0) return 0;
  const [min, max] = [Math.min(...values), Math.max(...values)];
  return max === min
    ? 50
    : attr === "l0_sparsity"
    ? 100 - ((value - min) / (max - min)) * 100
    : ((value - min) / (max - min)) * 100;
};

// 添加自定义样式组件
styled(TableCell)(({ theme }) => ({
  padding: "2px 20px",
  fontSize: "0.875rem",
  "&.MuiTableCell-head": {
    backgroundColor: theme.palette.grey[100],
    fontWeight: 600,
    position: "sticky",
    top: 0,
    zIndex: 1,
  },
  "&.column-layer": {
    width: "30px",
    maxWidth: "30px",
  },
  "&.column-type": {
    width: "50px",
    maxWidth: "50px",
  },
  "&.column-value": {
    width: "60px",
    maxWidth: "60px",
  },
  "&.column-rank": {
    width: "60px",
    maxWidth: "60px",
  },
  "&.column-contrib": {
    width: "180px",
    maxWidth: "180px",
    padding: "0 8px",
    height: "24px",
    "& > div": {
      width: "100%",
      overflowX: "auto",
      display: "flex",
      alignItems: "center",
      "&::-webkit-scrollbar": {
        height: "3px",
      },
      "&::-webkit-scrollbar-thumb": {
        backgroundColor: "rgba(0,0,0,.2)",
        borderRadius: "2px",
      },
    },
  },
}));
styled(TableRow)(({ theme }) => ({
  height: "16px",
  "& > *": {
    padding: "2px 20px",
    height: "16px",
  },
  "&:nth-of-type(odd)": {
    backgroundColor: theme.palette.action.hover,
  },
  "&:hover": {
    backgroundColor: theme.palette.action.selected,
  },
  "&.Mui-selected": {
    backgroundColor: theme.palette.primary.light,
    "&:hover": {
      backgroundColor: theme.palette.primary.light,
    },
  },
  cursor: "pointer",
}));
export 
const getDominantCluster = (points: ScatterPoint[]) => {
  const clusterCounts: { [key: number]: number } = {};
  const clusterColors: { [key: number]: string } = {};

  points.forEach((p) => {
    clusterCounts[p.clusterId] = (clusterCounts[p.clusterId] || 0) + 1;
    if (p.color) clusterColors[p.clusterId] = p.color;
  });

  const dominantCluster = Object.entries(clusterCounts).reduce((a, b) =>
    Number(a[1]) > Number(b[1]) ? a : b
  )[0];

  return {
    clusterId: Number(dominantCluster),
    color: clusterColors[Number(dominantCluster)] || "#ccc",
  };
};

export const addAlpha = (hexColor: string, alpha: number): string => {
  if (hexColor === undefined) return "#000";

  const r = parseInt(hexColor.slice(1, 3), 16);
  const g = parseInt(hexColor.slice(3, 5), 16);
  const b = parseInt(hexColor.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

// Make colors more vibrant/saturated - for bubble effect
export const saturateColor = (hexColor: string, factor: number = 1.4): string => {
  if (!hexColor || hexColor === undefined) return "#6366f1";

  // Handle rgb() format colors too
  let r: number, g: number, b: number;
  if (hexColor.startsWith('rgb')) {
    const match = hexColor.match(/\d+/g);
    if (!match) return "#6366f1";
    [r, g, b] = match.map(Number);
  } else if (hexColor.startsWith('#')) {
    r = parseInt(hexColor.slice(1, 3), 16);
    g = parseInt(hexColor.slice(3, 5), 16);
    b = parseInt(hexColor.slice(5, 7), 16);
  } else {
    return "#6366f1";
  }

  // Convert to HSL
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0, l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
    }
  }

  // BOOST saturation significantly - ensure minimum saturation for vibrancy
  s = Math.max(0.7, Math.min(1, s * factor));
  // Optimal lightness range for vibrant bubble colors
  l = Math.min(0.55, Math.max(0.4, l));

  // Convert back to RGB
  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const newR = Math.round(hue2rgb(p, q, h + 1/3) * 255);
  const newG = Math.round(hue2rgb(p, q, h) * 255);
  const newB = Math.round(hue2rgb(p, q, h - 1/3) * 255);

  return `rgb(${newR}, ${newG}, ${newB})`;
};

export function convertClusterData(level: string, clusters: any): ClusterData {
  const clusterData = clusters[level];
  return {
    clusterCount: parseInt(level),
    labels: clusterData.labels,
    colors: clusterData.colors.map((c: string) => c),
    centers: clusterData.centers,
    topics: clusterData.topics,
    topicScores: clusterData.topic_scores,
    clusterColors: clusterData.cluster_colors || {},
  };
}

export const createBubbleSet = (
  svgRef: React.RefObject<SVGSVGElement | null>,
  xScale: d3.ScaleLinear<number, number>,
  yScale: d3.ScaleLinear<number, number>,
  relatedPoints: ScatterPoint[],
  scale: number
) => {
  if (!svgRef.current || relatedPoints.length === 0) return;

  const g = d3.select(svgRef.current).select("g");

  g.selectAll(".bubble-set-path").remove();

  if (relatedPoints.length === 0) return;

  const pointSize = 4 / scale;
  const rectangles = relatedPoints.map((point) => ({
    x: xScale(point.x) - pointSize,
    y: yScale(point.y) - pointSize,
    width: pointSize * 2,
    height: pointSize * 2,
  }));

  const bubbles = new BubbleSet();
  const pad = 8 / scale;

  const list = bubbles.createOutline(
    BubbleSet.addPadding(rectangles, pad),
    [],
    null 
  );

  const outline = new PointPath(list).transform([
    new ShapeSimplifier(0.0),
    new BSplineShapeGenerator(),
    new ShapeSimplifier(0.0),
  ]);

  g.append("path")
    .attr("class", "bubble-set-path")
    .attr("d", `${outline}`)
    .style("fill", "rgba(255, 102, 0, 0.15)") 
    .style("stroke", "rgba(255, 102, 0, 0.5)") 
    .style("stroke-width", 1 / scale) 
    .style("pointer-events", "none"); 
};

export const updateHexbins = (
  svgRef: React.RefObject<SVGSVGElement | null>,
  dataRef: React.RefObject<ScatterPoint[]>,
  xScale: React.RefObject<d3.ScaleLinear<number, number>>,
  yScale: React.RefObject<d3.ScaleLinear<number, number>>,
  tooltipRef: React.RefObject<
    d3.Selection<HTMLDivElement, unknown, HTMLElement, any>
  >,
  selectedClusterRef: React.RefObject<{ level: string; data: ClusterData }>,
  scale: number,
  handlePointClick?: (point: ScatterPoint) => void
) => {
  if (!svgRef.current || !xScale.current || !yScale.current) return;

  console.log("updateHexbins", dataRef.current);

  const points = dataRef.current;
  const width = svgRef.current.clientWidth;
  const height = svgRef.current.clientHeight;

  const x = xScale.current;
  const y = yScale.current;

  const g = d3.select(svgRef.current).select("g");

  // Clear the previously drawn graphics
  g.selectAll(".token-related-point").remove();

  // Clear the previous fixed tooltip
  g.selectAll(".fixed-tooltip").remove();

  // Calculate the min and max similarity of similar points for normalization
  const similarPoints = points.filter(
    (p) => p.isQuerySimilar && p.similarity !== undefined
  );
  let minSimilarity = 1;
  let maxSimilarity = 0;

  if (similarPoints.length > 0) {
    minSimilarity =
      d3.min(similarPoints, (d) => d.similarity || 1) || minSimilarity;
    maxSimilarity =
      d3.max(similarPoints, (d) => d.similarity || 0) || maxSimilarity;
  }

  // Normalization function to map similarity to [0, 1]
  const normalizeSimilarity = (similarity: number | undefined): number => {
    if (similarity === undefined) return 0.5;
    // If min and max are equal, return 0.5 to avoid division by zero
    if (maxSimilarity === minSimilarity) return 0.5;
    return (similarity - minSimilarity) / (maxSimilarity - minSimilarity);
  };

  const hexbin = d3Hexbin
    .hexbin<ScatterPoint>()
    .x((d: ScatterPoint) => x(d.x))
    .y((d: ScatterPoint) => y(d.y))
    .radius(7)
    .extent([
      [0, 0],
      [width, height],
    ]);

  // Compute hexbin using all non-query points
  const allPoints = points.filter((p) => !p.isQuery);
  const bins = hexbin(allPoints);

  // REMOVED: Hexagonal cluster backgrounds - user wants circles only
  // Clear any existing hexagons
  g.selectAll(".hexagon").remove();

  const maxRelatedTokenLength = Math.max(
    ...points.map((p) => p.relatedToken?.length || 0)
  );

  // Show ALL points as circles (not just visible ones)
  const normalPoints = g
    .selectAll<SVGCircleElement, ScatterPoint>(".point")
    .data(
      points
        .filter((p) => {
          if (p.isQuery) return false;
          // Don't filter by intersection points here - show all as circles
          if (p.relatedToken && p.relatedToken.length >= maxRelatedTokenLength)
            return false;
          // CHANGED: Show ALL points, not just visible ones
          return true;
        })
        .sort((a, b) => {
          if (a.isQuerySimilar && b.isQuerySimilar) {
            return (a.similarity || 0) - (b.similarity || 0);
          }
          if (a.isQuerySimilar && !b.isQuerySimilar) return 1;
          if (!a.isQuerySimilar && b.isQuerySimilar) return -1;
          return 0;
        }),
      (d) => `${d.index}`
    );

  normalPoints.exit().remove();

  const normalPointsEnter = normalPoints
    .enter()
    .append("circle")
    .attr("class", "point")
    .style("fill", (d) => {
      if (d.isQuerySimilar) {
        const normalizedSimilarity = normalizeSimilarity(d.similarity);
        const enhancedSimilarity = Math.pow(normalizedSimilarity, 0.8);

        let mappedValue;
        if (enhancedSimilarity >= 0.6) {
          mappedValue = 0.7 + (enhancedSimilarity - 0.6) * 0.75;
        } else if (enhancedSimilarity >= 0.4) {
          mappedValue = 0.2 + (enhancedSimilarity - 0.4) * 2.5;
        } else {
          mappedValue = enhancedSimilarity * 0.5;
        }

        return interpolateColor('#7070b0', '#1e78ff', mappedValue);
      }

      // Open ball style - light fill with visible color
      return d.color ? addAlpha(d.color, 0.3) : "rgba(99, 102, 241, 0.3)";
    })
    .attr("cx", (d) => x(d.x))
    .attr("cy", (d) => y(d.y))
    .attr("r", 0)
    .style("stroke", (d) => d.color || "#6366f1")  // Solid colored border
    .style("stroke-width", 2 / scale)
    .style("stroke-opacity", 1);

  function interpolateColor(color1:any, color2:any, factor:any) {
    const parseColor = (color:any) => {
      const hex = color.replace('#', '');
      return {
        r: parseInt(hex.substr(0, 2), 16),
        g: parseInt(hex.substr(2, 2), 16),
        b: parseInt(hex.substr(4, 2), 16)
      };
    };
    
    const c1 = parseColor(color1);
    const c2 = parseColor(color2);
    
    const r = Math.round(c1.r + (c2.r - c1.r) * factor);
    const g = Math.round(c1.g + (c2.g - c1.g) * factor);
    const b = Math.round(c1.b + (c2.b - c1.b) * factor);
    
    return `rgb(${r}, ${g}, ${b})`;
  }

  const top5SimilarPoints = (() => {
    const sortedPoints = [...similarPoints].sort(
      (a, b) => (b.similarity || 0) - (a.similarity || 0)
    );

    const top5 = sortedPoints.slice(0, 5);

    const sampledPoints = [];
    for (let i = 15; i < Math.min(sortedPoints.length, 100); i += 15) {
      sampledPoints.push(sortedPoints[i]);
    }

    return [...top5, ...sampledPoints];
  })();

  const centerX =
    top5SimilarPoints.slice(0, 5).reduce((sum, p) => sum + x(p.x), 0) / 5;
  const centerY =
    top5SimilarPoints.slice(0, 5).reduce((sum, p) => sum + y(p.y), 0) / 5;

  [...top5SimilarPoints].reverse().forEach((point, reversedIndex) => {
    const index = top5SimilarPoints.length - 1 - reversedIndex;

    let rank;
    if (index < 5) {
      rank = index + 1;
    } else {
      rank = (index - 5) * 15 + 16; 
    }

    const vectorX = (x(point.x) - centerX) ;
    const vectorY = (y(point.y) - centerY) ;

    const length = Math.sqrt(vectorX * vectorX + vectorY * vectorY);
    const normalizedX = vectorX / (length || 1); 
    const normalizedY = vectorY / (length || 1);

    let distance = 30; 

    const tooltipX = x(point.x) + normalizedX * distance;
    const tooltipY = y(point.y) + normalizedY * distance;

    const rectWidth = 200 / scale;
    const rectHeight = 80 / scale;
    const idFontSize = 10 / scale;
    const descriptionFontSize = 11 / scale;
    const rankCircleRadius = 8 / scale;
    const padding = 3 / scale;

    const minScale = 2;
    const maxScale = 8;

    // Calculate the current transparency (linear interpolation)
    let opacity = (scale - minScale) / (maxScale - minScale);
    opacity = Math.max(0, Math.min(1, opacity)); // Limit to 0-1 range

    // If opacity is 0, skip rendering to improve performance
    if (opacity <= 0) return;

    // Determine the connection point of the line on the tooltip
    // Determine the connection point based on the vector direction
    let connectorX, connectorY;

    // Determine the main direction of the tooltip relative to the point
    // Divide 360 degrees into four quadrants to determine the direction of the tooltip
    const angle = (Math.atan2(normalizedY, normalizedX) * 180) / Math.PI;

    if (angle >= -45 && angle < 45) {
      connectorX = 0;
      connectorY = rectHeight / 2;
    } else if (angle >= 45 && angle < 135) {
      connectorX = rectWidth / 2;
      connectorY = 0;
    } else if (
      (angle >= 135 && angle <= 180) ||
      (angle >= -180 && angle < -135)
    ) {
      connectorX = rectWidth;
      connectorY = rectHeight / 2;
    } else {
      connectorX = rectWidth / 2;
      connectorY = rectHeight;
    }

    const fixedTooltip = g
      .append("g")
      .attr("class", "fixed-tooltip")
      .attr("transform", `translate(${tooltipX}, ${tooltipY})`)
      .style("opacity", opacity);

    // Text processing function: split long text into two lines and add ellipsis
    const formatDescription = (
      text: string
    ): { line1: string; line2: string } => {
      if (!text) return { line1: "", line2: "" };

      const charsPerLine = Math.floor(rectWidth / (descriptionFontSize * 0.6));

      if (text.length <= charsPerLine) {
        return { line1: text, line2: "" };
      } else if (text.length <= charsPerLine * 2) {
        const breakPoint = text.lastIndexOf(" ", charsPerLine);
        const splitIndex =
          breakPoint > charsPerLine / 2 ? breakPoint : charsPerLine;
        return {
          line1: text.substring(0, splitIndex),
          line2: text.substring(splitIndex).trim(),
        };
      } else {
        const breakPoint = text.lastIndexOf(" ", charsPerLine);
        const splitIndex =
          breakPoint > charsPerLine / 2 ? breakPoint : charsPerLine;
        const secondLineLength = charsPerLine - 3; 

        return {
          line1: text.substring(0, splitIndex),
          line2:
            text.substring(splitIndex, splitIndex + secondLineLength).trim() +
            "...",
        };
      }
    };

    // Create tooltip background
    fixedTooltip
      .append("rect")
      .attr("rx", 6 / scale)
      .attr("ry", 6 / scale)
      .attr("width", rectWidth)
      .attr("height", rectHeight)
      .attr("fill", "rgba(255, 255, 255, 0.9)")
      .attr("stroke", "#1976d2")
      .attr("stroke-width", 1 / scale)
      .attr("stroke-opacity", 0.5)
      .style("filter", "drop-shadow(0px 1px 2px rgba(0,0,0,0.2))");

    // Add rank marker (top left corner)
    fixedTooltip
      .append("circle")
      .attr("cx", 10 / scale)
      .attr("cy", 10 / scale)
      .attr("r", rankCircleRadius)
      .attr("fill", "#1976d2");

    fixedTooltip
      .append("text")
      .attr("x", 10 / scale)
      .attr("y", 13 / scale)
      .attr("text-anchor", "middle")
      .attr("font-size", idFontSize)
      .attr("font-weight", "bold")
      .attr("fill", "white")
      .text(rank);

    // Add ID information (compact display at the top)
    fixedTooltip
      .append("text")
      .attr("x", 22 / scale)
      .attr("y", 13 / scale)
      .attr("font-size", idFontSize)
      .attr("font-weight", "bold")
      .attr("fill", "#1976d2")
      .text(`ID: ${point.featureId}`);

    // Add similarity information (top right corner)
    fixedTooltip
      .append("text")
      .attr("x", rectWidth - padding)
      .attr("y", 13 / scale)
      .attr("font-size", idFontSize)
      .attr("font-weight", "bold")
      .attr("fill", "#1976d2")
      .attr("text-anchor", "end")
      .text(`Sim: ${point.similarity?.toFixed(3)}`);

    // Add separator line
    fixedTooltip
      .append("line")
      .attr("x1", padding)
      .attr("y1", 18 / scale)
      .attr("x2", rectWidth - padding)
      .attr("y2", 18 / scale)
      .attr("stroke", "rgba(25, 118, 210, 0.2)")
      .attr("stroke-width", 1 / scale);

    // Process the description text
    const descriptionText = point.description || "";
    const formattedDesc = formatDescription(descriptionText);

    // Add description text (main content display) - line 1
    fixedTooltip
      .append("text")
      .attr("x", 10 / scale)
      .attr("y", 35 / scale)
      .attr("font-size", descriptionFontSize)
      .attr("font-weight", "medium")
      .attr("fill", "#333")
      .text(formattedDesc.line1);

    // Add description text - line 2 (if exists)
    if (formattedDesc.line2) {
      fixedTooltip
        .append("text")
        .attr("x", 10 / scale)
        .attr("y", 55 / scale)
        .attr("font-size", descriptionFontSize)
        .attr("font-weight", "medium")
        .attr("fill", "#333")
        .text(formattedDesc.line2);
    }

    // Add connection line from the point to the tooltip
    g.append("line")
      .attr("class", "fixed-tooltip")
      .attr("x1", x(point.x))
      .attr("y1", y(point.y))
      .attr("x2", tooltipX + connectorX)
      .attr("y2", tooltipY + connectorY)
      .attr("stroke", "#1976d2")
      .attr("stroke-width", 1 / scale)
      .attr("stroke-dasharray", `${3 / scale},${3 / scale}`)
      .attr("stroke-opacity", opacity * 0.5); // The transparency of the connection line is consistent with the tooltip
  });

  normalPointsEnter
    .on("mouseover", (event, d) => {
      const similarityText = d.similarity
        ? `Similarity: ${d.similarity.toFixed(3)}`
        : "";

      tooltipRef.current?.style("visibility", "visible").html(`
        <div style="font-family: system-ui, -apple-system, sans-serif;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-weight: 600; color: #1976d2; font-size: 14px;">Feature ID: ${
              d.featureId
            }</span>
            ${
              d.similarity
                ? `<span style="background: rgba(25, 118, 210, 0.1); color: #1976d2; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">${similarityText}</span>`
                : ""
            }
          </div>
          ${
            d.description
              ? `
          <div style="margin-top: 5px; max-width: 280px; font-size: 13px; line-height: 1.4; color: #555;">
            ${d.description}
          </div>`
              : ""
          }
          ${
            d.relatedToken
              ? `
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
            <span style="font-size: 12px; color: #ff5722; font-weight: 500;">Related tokens: ${d.relatedToken.length}</span>
          </div>`
              : ""
          }
        </div>
      `);
    })
    .on("mousemove", (event) => {
      tooltipRef.current
        ?.style("top", `${event.pageY - 10}px`)
        ?.style("left", `${event.pageX + 10}px`);
    })
    .on("mouseout", () => tooltipRef.current?.style("visibility", "hidden"));

  if (handlePointClick) {
    normalPointsEnter.on("click", (event, d) => handlePointClick(d));
  }

  normalPointsEnter
  .merge(normalPoints)
  .transition()
  .duration(300)
  .attr("r", (d) => {
    // Make concept-highlighted or search-highlighted points larger
    if (d.isConceptHighlighted || d.isSearchHighlighted) return 8 / scale;
    return 6 / scale;
  })
  .attr("cx", (d) => x(d.x))
  .attr("cy", (d) => y(d.y))
  .style("fill", (d) => {
    // Concept highlighted points get a bright red fill
    if (d.isConceptHighlighted) {
      return "rgba(239, 68, 68, 0.6)"; // Red fill for concept highlighted
    }
    // Search highlighted points get a bright orange fill
    if (d.isSearchHighlighted) {
      return "rgba(249, 115, 22, 0.6)"; // Orange fill for search highlighted
    }

    if (d.isQuerySimilar) {
      const normalizedSimilarity = normalizeSimilarity(d.similarity);
      const enhancedSimilarity = Math.pow(normalizedSimilarity, 0.8);

      let mappedValue;
      if (enhancedSimilarity >= 0.6) {
        mappedValue = 0.7 + (enhancedSimilarity - 0.6) * 0.75;
      } else if (enhancedSimilarity >= 0.4) {
        mappedValue = 0.2 + (enhancedSimilarity - 0.4) * 2.5;
      } else {
        mappedValue = enhancedSimilarity * 0.5;
      }

      // Blue gradient based on similarity
      const lowColor = { r: 100, g: 100, b: 180 };
      const highColor = { r: 30, g: 120, b: 255 };

      const r = Math.round(lowColor.r + (highColor.r - lowColor.r) * mappedValue);
      const g = Math.round(lowColor.g + (highColor.g - lowColor.g) * mappedValue);
      const b = Math.round(lowColor.b + (highColor.b - lowColor.b) * mappedValue);

      return `rgb(${r}, ${g}, ${b})`;
    }

    // Open ball style - semi-transparent fill
    return d.color ? addAlpha(d.color, 0.25) : "rgba(99, 102, 241, 0.25)";
  })
  .style("stroke", (d) => {
    // Concept highlighted points get red border
    if (d.isConceptHighlighted) return "#ef4444";
    // Search highlighted points get orange border
    if (d.isSearchHighlighted) return "#f97316";
    if (d.isVisibleInPanel) return "#ef4444";
    return d.color || "#6366f1";
  })
  .style("stroke-width", (d) => {
    if (d.isConceptHighlighted || d.isSearchHighlighted) return 3 / scale;
    return d.isVisibleInPanel ? 3 / scale : 2 / scale;
  })
  .style("stroke-opacity", 1)
  .style("fill-opacity", (d) => {
    // Dim non-highlighted points when concept or search is active
    // We detect if any point is highlighted to know if highlighting is active
    const hasHighlightedPoints = points.some(p => p.isConceptHighlighted || p.isSearchHighlighted);
    if (hasHighlightedPoints) {
      return (d.isConceptHighlighted || d.isSearchHighlighted) ? 0.8 : 0.15;
    }
    return 0.4;
  });

  normalPoints
    .merge(normalPointsEnter)
    .on("mouseover", function (event, d) {
      const similarityText = d.similarity
        ? `Similarity: ${d.similarity.toFixed(3)}`
        : "";

      // Enhanced tooltip with Feature ID, Layer, and Cluster info
      tooltipRef.current?.style("visibility", "visible").html(`
        <div style="font-family: system-ui, -apple-system, sans-serif;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-weight: 600; color: #1976d2; font-size: 14px;">Feature ${d.featureId}</span>
            ${
              d.similarity
                ? `<span style="background: rgba(25, 118, 210, 0.1); color: #1976d2; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">${similarityText}</span>`
                : ""
            }
          </div>
          <div style="display: flex; gap: 8px; margin-bottom: 5px;">
            ${d.layer ? `<span style="background: rgba(34, 197, 94, 0.1); color: #22c55e; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500;">Layer ${d.layer}</span>` : ""}
            <span style="background: rgba(168, 85, 247, 0.1); color: #a855f7; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500;">Cluster ${d.clusterId}</span>
          </div>
          ${
            d.description
              ? `
          <div style="margin-top: 5px; max-width: 280px; font-size: 13px; line-height: 1.4; color: #555;">
            ${d.description}
          </div>`
              : ""
          }
          ${
            d.relatedToken
              ? `
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
            <span style="font-size: 12px; color: #ff5722; font-weight: 500;">Related tokens: ${d.relatedToken.length}</span>
          </div>`
              : ""
          }
        </div>
      `);

      d3.select(this)
        .transition()
        .duration(150)
        .attr("r", 10 / scale)  // Larger on hover
        .style("stroke", "#1976d2")
        .style("stroke-width", 3 / scale)
        .style("stroke-opacity", 1);
    })
    .on("mousemove", (event) => {
      tooltipRef.current
        ?.style("top", `${event.pageY - 10}px`)
        ?.style("left", `${event.pageX + 10}px`);
    })
    .on("mouseout", function () {
      tooltipRef.current?.style("visibility", "hidden");

      d3.select(this)
        .transition()
        .duration(150)
        .attr("r", 6 / scale)  // Return to normal size
        .style("stroke", function (d) {
          const point = d as unknown as ScatterPoint;
          return point.isVisibleInPanel ? "#ef4444" : point.color || "#6366f1";
        })
        .style("stroke-width", function (d) {
          const point = d as unknown as ScatterPoint;
          return point.isVisibleInPanel ? 3 / scale : 2 / scale;
        })
        .style("stroke-opacity", 1);
    });

  const relatedPoints = points.filter(
    (p) => p.relatedToken && p.relatedToken.length >= maxRelatedTokenLength
  );

  // Draw the intersection point (changed to triangle)
  const intersectionSelection = g
    .selectAll<SVGPathElement, ScatterPoint>(".intersection-point")
    .data(
      relatedPoints.filter(
        (p) => p.relatedToken && p.relatedToken.length >= maxRelatedTokenLength
      ),
      (d) => `${d.index}-intersection`
    );

  intersectionSelection.exit().remove();

  const triangleSize = 4 / scale;
  const trianglePath = (d: ScatterPoint) => {
    const pointX = x(d.x);
    const pointY = y(d.y);
    return `M ${pointX} ${pointY - triangleSize} L ${pointX + triangleSize} ${
      pointY + triangleSize
    } L ${pointX - triangleSize} ${pointY + triangleSize} Z`;
  };

  const intersectionEnter = intersectionSelection
    .enter()
    .append("path")
    .attr("class", "token-related-point intersection-point")
    .attr("d", trianglePath);

  // Add mouse hover effect to intersection
  intersectionEnter
    .on("mouseover", function (event, d) {
      // Format activation values
      const activationValues = d.relatedToken
        ? d.relatedToken
            .map(
              (token) =>
                `${token.prompt.substring(0, 15)}...[${
                  token.index
                }]: ${token.activation.toFixed(3)}`
            )
            .join("<br/>")
        : "";

      tooltipRef.current?.style("visibility", "visible").html(`
        <div style="font-family: system-ui, -apple-system, sans-serif;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #ff5722; font-size: 14px;">Intersection Feature ID: ${
              d.featureId
            }</span>
            <span style="background: rgba(255, 87, 34, 0.1); color: #ff5722; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">
              Related tokens: ${d.relatedToken?.length || 0}
            </span>
          </div>
          ${
            d.description
              ? `
          <div style="margin-top: 5px; max-width: 280px; font-size: 13px; line-height: 1.4; color: #555;">
            ${d.description}
          </div>`
              : ""
          }
          ${
            d.relatedToken
              ? `
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
            <span style="font-size: 12px; color: #555; font-weight: 500;">Activation details:</span>
            <div style="margin-top: 5px; font-size: 12px; color: #666; line-height: 1.5;">
              ${activationValues}
            </div>
          </div>`
              : ""
          }
        </div>
      `);

      d3.select(this)
        .transition()
        .duration(150)
        .style("fill", "#ff7043")
        .style("stroke-width", 2 / scale)
        .attr(
          "transform",
          `translate(${x(d.x)}, ${y(d.y)}) scale(1.3) translate(${-x(
            d.x
          )}, ${-y(d.y)})`
        );
    })
    .on("mousemove", (event) => {
      tooltipRef.current
        ?.style("top", `${event.pageY - 10}px`)
        ?.style("left", `${event.pageX + 10}px`);
    })
    .on("mouseout", function () {
      tooltipRef.current?.style("visibility", "hidden");

      d3.select(this)
        .transition()
        .duration(150)
        .style("fill", "#ff5722")
        .style("stroke-width", function (d) {
          const point = d as unknown as ScatterPoint;
          return point.isVisibleInPanel ? 2 / scale : 1 / scale;
        })
        .attr("transform", "scale(1) translate(0, 0)");
    });

  if (handlePointClick) {
    intersectionEnter.on("click", (event, d) => handlePointClick(d));
  }

  intersectionEnter
    .merge(intersectionSelection)
    .attr("d", trianglePath)
    .style("fill", "#ff5722")
    .style("stroke", "#ff5722")
    .style("stroke-width", (d) => (d.isVisibleInPanel ? 2 / scale : 1 / scale));

  const queryPoints = g
    .selectAll<SVGGElement, ScatterPoint>(".query-point-group")
    .data(
      points.filter((p) => p.isQuery),
      (d) => `${d.index}-query`
    );

  queryPoints.exit().remove();

  const queryPointsEnter = queryPoints
    .enter()
    .append("g")
    .attr("class", "query-point-group");

  const allQueryPoints = queryPointsEnter
    .merge(queryPoints)
    .attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`);

  allQueryPoints.selectAll("*").remove();

  // Pushpin size
  const pinHeadRadius = 8 / scale; // Pushpin head radius
  const pinLength = 24 / scale; // Pushpin needle length
  const pinTipWidth = 2 / scale; // Pushpin tip width

  // Draw the pin part (the cone from the head to the tip)
  allQueryPoints
    .append("path")
    .attr("d", (d) => {
      return `
        M 0 0
        L 0 -${pinLength}
        L -${pinTipWidth / 2} -${pinLength * 0.7}
        L 0 -${pinLength * 0.5}
        L ${pinTipWidth / 2} -${pinLength * 0.7}
        L 0 -${pinLength}
        Z
      `;
    })
    .style("fill", "#666")
    .style("stroke", "#333")
    .style("stroke-width", 0.5 / scale);

  // Draw the pin head (circular)
  allQueryPoints
    .append("circle")
    .attr("cx", 0)
    .attr("cy", -pinLength)
    .attr("r", pinHeadRadius)
    .style("fill", "rgb(225, 25, 25)") 
    .style("stroke", "#880000")
    .style("stroke-width", 1 / scale)
    .style("filter", "drop-shadow(1px 1px 1px rgba(0,0,0,0.3))");

  // After drawing all points, add BubbleSet
  if (relatedPoints.length > 0 && xScale.current && yScale.current) {
    createBubbleSet(
      svgRef,
      xScale.current,
      yScale.current,
      relatedPoints,
      scale
    );
  }

  // Draw the cluster center topic labels
  // First remove existing labels
  g.selectAll(".cluster-topic-label").remove();

  // Check if there is cluster data and the scale is suitable for displaying labels (do not display labels when below a certain threshold to avoid crowding)
  if (selectedClusterRef.current?.data && scale > 0.8) {
    const clusterData = selectedClusterRef.current.data;

    // Iterate over cluster centers and add labels
    Object.keys(clusterData.topics).forEach((clusterId) => {
      const id = parseInt(clusterId);
      // Ensure the cluster has a center point and topics
      if (
        clusterData.centers[id] &&
        clusterData.topics[clusterId] &&
        clusterData.topics[clusterId].length > 0
      ) {
        const center = clusterData.centers[id];
        const topics = clusterData.topics[clusterId];
        const clusterColor = clusterData.clusterColors[clusterId] || "#333";

        const bgPadding = 4 / scale;

        const fontSize = Math.min(16, Math.max(14, 16 * scale)) / scale;
        const topicsToShow = scale > 2.5 ? 4 : scale > 1.5 ? 3 : 2;

        const labelGroup = g
          .append("g")
          .attr("class", "cluster-topic-label")
          .attr("transform", `translate(${x(center[0])},${y(center[1])})`);

        const textContainer = labelGroup.append("g");

        const mainTopicElement = textContainer
          .append("text")
          .attr("class", "main-topic")
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("y", 0)
          .attr("font-size", `${fontSize}px`)
          .attr("font-weight", "bold")
          .attr("fill", clusterColor)
          .attr("paint-order", "stroke")
          .attr("stroke", "white")
          .attr("stroke-width", 1 / scale)
          .text(topics[0]);

        const secondaryTextElements = [];
        let maxY = 0;

        if (topicsToShow > 1 && topics.length > 1) {
          const mainTextBBox = mainTopicElement.node()?.getBBox();
          let yOffset = (mainTextBBox?.height || 0) * 0.6 + 0.1;

          for (let i = 1; i < Math.min(topicsToShow, topics.length); i++) {
            const secondaryFontSize = fontSize * 0.9;

            const secondaryElement = textContainer
              .append("text")
              .attr("class", "secondary-topic")
              .attr("text-anchor", "middle")
              .attr("dominant-baseline", "middle")
              .attr("y", yOffset)
              .attr("font-size", `${secondaryFontSize}px`)
              .attr("fill", clusterColor)
              .attr("paint-order", "stroke")
              .attr("stroke", "white")
              .attr("stroke-width", 0.5 / scale)
              .text(topics[i]);

            secondaryTextElements.push(secondaryElement);

            const secondaryBBox = secondaryElement.node()?.getBBox();
            yOffset += (secondaryBBox?.height || 0) * 0.9;
            maxY = yOffset;
          }
        }

        const textNodes = textContainer
          .selectAll<SVGTextElement, unknown>("text")
          .nodes();
        if (textNodes.length > 0) {
          let minX = Infinity,
            minY = Infinity,
            maxX = -Infinity,
            maxYValue = -Infinity;

          textNodes.forEach((node) => {
            if (node) {
              const bbox = node.getBBox();
              minX = Math.min(minX, bbox.x);
              minY = Math.min(minY, bbox.y);
              maxX = Math.max(maxX, bbox.x + bbox.width);
              maxYValue = Math.max(maxYValue, bbox.y + bbox.height);
            }
          });

          labelGroup
            .insert("rect", "g")
            .attr("x", minX - bgPadding)
            .attr("y", minY - bgPadding)
            .attr("width", maxX - minX + 2 * bgPadding)
            .attr("height", maxYValue - minY + 2 * bgPadding)
            .attr("rx", 3 / scale)
            .attr("ry", 3 / scale)
            .attr("fill", "rgba(255, 255, 255, 0.7)");
        }
      }
    });
  }
};

export const initZoom = (
  svgRef: React.RefObject<SVGSVGElement | null>,
  dataRef: React.RefObject<ScatterPoint[]>,
  transformRef: React.RefObject<d3.ZoomTransform>,
  zoomRef: React.RefObject<d3.ZoomBehavior<SVGSVGElement, unknown>>,
  updateHexbins: (scale: number) => void,
  hierarchicalClusters: any,
  selectedClusterRef: React.RefObject<{
    level: string;
    data: any;
  }>,
  zoomThresholds: {
    low: { zoom: number; level: string };
    medium: { zoom: number; level: string };
    high: { zoom: number; level: string };
  }
) => {
  if (!svgRef.current) return;

  let previousZoomLevel = 1; 

  const zoom = d3
    .zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.5, 8])
    .on("zoom", (event) => {
      const transform = event.transform;
      transformRef.current = transform;
      const g = d3.select(svgRef.current).select("g");

      g.attr("transform", transform);

      const currentZoom = transform.k;

      const zoomChanged = Math.abs(currentZoom - previousZoomLevel) > 0.001; 

      let newLevel = zoomThresholds.low.level;
      if (currentZoom >= zoomThresholds.high.zoom) {
        newLevel = zoomThresholds.high.level;
      } else if (currentZoom >= zoomThresholds.medium.zoom) {
        newLevel = zoomThresholds.medium.level;
      }

      const levelChanged = newLevel !== selectedClusterRef.current.level;

      if (zoomChanged || levelChanged) {
        if (levelChanged) {
          const updatedPoints = dataRef.current.map((point) =>
            point.isQuery
              ? point
              : {
                  ...point,
                  clusterId:
                    hierarchicalClusters[newLevel]?.labels[point.index] ?? -1,
                  color: hierarchicalClusters[newLevel]?.colors[point.index],
                }
          );

          dataRef.current = updatedPoints;
          selectedClusterRef.current = {
            level: newLevel,
            data: convertClusterData(newLevel, hierarchicalClusters),
          };
        }

        updateHexbins(transform.k);

        previousZoomLevel = currentZoom;
      }
    });

  zoomRef.current = zoom;
  d3.select(svgRef.current).call(zoom);
  return zoom;
};

export const initD3Chart = (
  svgRef: React.RefObject<SVGSVGElement | null>,
  dataRef: React.RefObject<ScatterPoint[]>,
  xScale: React.RefObject<d3.ScaleLinear<number, number>>,
  yScale: React.RefObject<d3.ScaleLinear<number, number>>,
  tooltipRef: React.RefObject<
    d3.Selection<HTMLDivElement, unknown, HTMLElement, any>
  >,
  transformRef: React.RefObject<d3.ZoomTransform>,
  zoomRef: React.RefObject<d3.ZoomBehavior<SVGSVGElement, unknown>>,
  updateHexbins: (scale: number) => void,
  initZoomFn: typeof initZoom,
  hierarchicalClusters: any,
  selectedClusterRef: React.RefObject<{
    level: string;
    data: any;
  }>,
  zoomThresholds: {
    low: { zoom: number; level: string };
    medium: { zoom: number; level: string };
    high: { zoom: number; level: string };
  }
) => {
  if (!svgRef.current) return;

  d3.select(svgRef.current).selectAll("*").remove();
  tooltipRef.current?.remove();

  const width = svgRef.current.clientWidth;
  const height = svgRef.current.clientHeight;

  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height);

  const points = dataRef.current;
  const xExtent = d3.extent(points, (d) => d.x) as [number, number];
  const yExtent = d3.extent(points, (d) => d.y) as [number, number];

  const x = d3.scaleLinear().domain(xExtent).range([0, width]);
  const y = d3.scaleLinear().domain(yExtent).range([height, 0]);
  xScale.current = x;
  yScale.current = y;
  svg.append("g");

  tooltipRef.current = d3
    .select("body")
    .append("div")
    .attr("class", "scatter-tooltip")
    .style("position", "absolute")
    .style("visibility", "hidden")
    .style("background-color", "rgba(255, 255, 255, 0.95)")
    .style("padding", "8px 12px")
    .style("border", "none")
    .style("border-radius", "6px")
    .style("box-shadow", "0 2px 8px rgba(0, 0, 0, 0.15)")
    .style("font-size", "14px")
    .style("line-height", "1.4")
    .style("color", "#333")
    .style("pointer-events", "none")
    .style("transition", "all 0.15s ease")
    .style("max-width", "280px");

  const zoom = initZoomFn(
    svgRef,
    dataRef,
    transformRef,
    zoomRef,
    updateHexbins,
    hierarchicalClusters,
    selectedClusterRef,
    zoomThresholds
  );

  const initialTransform = d3.zoomIdentity;
  if (zoom) {
    d3.select(svgRef.current).call(zoom).call(zoom.transform, initialTransform);
  }

  updateHexbins(initialTransform.k);
};

// Helper to get concept from description
export const extractPrimaryConcept = (description: string): string | undefined => {
  if (!description) return undefined;
  const descLower = description.toLowerCase();
  // Check for known concepts in order of specificity
  const concepts = Object.keys(CONCEPT_TYPE_MAP);
  for (const concept of concepts) {
    if (descLower.includes(concept.replace('_', ' ')) || descLower.includes(concept)) {
      return concept;
    }
  }
  return undefined;
};

export type ColoringMode = "cluster" | "concept" | "subconcept" | "layer";

export const createMainPoints = (
  data: any,
  visibleCoordinates: Set<string>,
  selectedClusterRef: React.RefObject<{ level: string; data: ClusterData }>,
  saeId?: string,  // Optional layer identifier from SAE ID (e.g., "action_expert_layer_12")
  coloringMode: ColoringMode = "cluster"  // Coloring mode: cluster, concept (type), subconcept (individual), layer
) => {
  const clusterLevel = selectedClusterRef.current?.level;
  if (!clusterLevel) {
    console.warn("selectedClusterRef not initialized yet.");
    return [];
  }

  // Extract layer number from SAE ID (e.g., "action_expert_layer_12-concepts" -> "12")
  let layer: string | undefined;
  const isAllLayers = saeId?.includes('all_layers');
  if (saeId && !isAllLayers) {
    const layerMatch = saeId.match(/layer_(\d+)/);
    if (layerMatch) {
      layer = layerMatch[1];
    }
  }

  return data.coordinates.map((coord: any, index: number) => {
    const rawIndex = data.indices[index];
    let featureId: number;
    let pointLayer: string | undefined = layer;

    // Handle all_layers compound indices like "0_10848" (layerNum_featureId)
    if (isAllLayers && typeof rawIndex === 'string' && rawIndex.includes('_')) {
      const parts = rawIndex.split('_');
      pointLayer = parts[0];  // Layer number
      featureId = parseInt(parts[1]);  // Actual feature ID
    } else {
      featureId = parseInt(rawIndex);
    }

    // Get description for concept extraction
    const description = data.descriptions[index];

    // Determine color based on coloringMode
    let color: string;
    if (coloringMode === "layer" && pointLayer !== undefined) {
      // Layer-based coloring
      color = LAYER_COLORS[pointLayer] || "#6366f1";
    } else if (coloringMode === "subconcept") {
      // Subconcept-based coloring - individual colors for each concept (PUT, OPEN, BOWL, etc.)
      const concept = extractPrimaryConcept(description);
      color = concept ? (SUBCONCEPT_COLORS[concept] || CONCEPT_TYPE_COLORS.unknown) : CONCEPT_TYPE_COLORS.unknown;
    } else if (coloringMode === "concept") {
      // Concept type-based coloring - colors by category (motion, object, phase, spatial)
      const concept = extractPrimaryConcept(description);
      const conceptType = concept ? CONCEPT_TYPE_MAP[concept] : "unknown";
      color = CONCEPT_TYPE_COLORS[conceptType] || CONCEPT_TYPE_COLORS.unknown;
    } else {
      // Default: cluster-based coloring
      color = data.hierarchical_clusters[clusterLevel]?.colors[index] || "#6366f1";
    }

    return {
      x: coord[0],
      y: coord[1],
      index,
      featureId,
      description,
      layer: pointLayer, // Include layer info in each point
      clusterId: data.hierarchical_clusters[clusterLevel]?.labels[index] ?? -1,
      color,
      visible: visibleCoordinates.has(coord.join(",")),
    };
  });
};

export const createQueryPoint = (
  data: any,
  selectedClusterRef: React.RefObject<{ level: string; data: ClusterData }>
) => {
  const clusterLevel = selectedClusterRef.current?.level;
  if (!clusterLevel) {
    console.warn("selectedClusterRef not initialized yet.");
    return null; 
  }

  return data.query
    ? {
        x: data.query.coordinates[0],
        y: data.query.coordinates[1],
        index: 0,
        featureId: 0,
        description: data.query.text,
        isQuery: true,
        clusterId: -1,
      }
    : null;
};
