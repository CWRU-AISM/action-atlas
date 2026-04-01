"use client";
import React from "react";
import {ModelData, OverviewMatrixProps} from "@/types/types";

const OverviewMatrix: React.FC<OverviewMatrixProps> = ({modelData, visibleRange, calculateAverageRank,}) => {
  const type = "RES";

  const maxLayer = modelData.length > 0 ? Math.max(...modelData.map(model => model.layer)) : 25;
  const totalLayers = maxLayer + 1;

  const getValueColor = (model: ModelData | undefined) => {
    if (!model) return "rgb(255, 255, 255)";
    const avgRank = calculateAverageRank(model);

    const normalizedValue = (totalLayers - avgRank) / (totalLayers - 1);

    const y = normalizedValue >= 0.8 ? 0.8 + (normalizedValue - 0.8) * 1 : (normalizedValue >= 0.7 ? 0.1 + (normalizedValue - 0.7) * 7 : normalizedValue / 3);

    return interpolateColor('#fde3e3', '#c62828', y);
  };

  const interpolateColor = (colorA: any, colorB: any, value: number) => {
    const t = Math.max(0, Math.min(1, value));

    const parseColor = (color: any) => {
      if (typeof color === 'string') {
        if (color.startsWith('#')) {
          const hex = color.slice(1);
          if (hex.length === 3) {
            return {
              r: parseInt(hex[0] + hex[0], 16),
              g: parseInt(hex[1] + hex[1], 16),
              b: parseInt(hex[2] + hex[2], 16)
            };
          } else if (hex.length === 6) {
            return {
              r: parseInt(hex.slice(0, 2), 16),
              g: parseInt(hex.slice(2, 4), 16),
              b: parseInt(hex.slice(4, 6), 16)
            };
          }
        }

        const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (rgbMatch) {
          return {
            r: parseInt(rgbMatch[1]),
            g: parseInt(rgbMatch[2]),
            b: parseInt(rgbMatch[3])
          };
        }
      }

      if (typeof color === 'object' && color.r !== undefined) {
        return color;
      }

      return { r: 0, g: 0, b: 0 };
    };

    const rgbA = parseColor(colorA);
    const rgbB = parseColor(colorB);

    const r = Math.round(rgbA.r + (rgbB.r - rgbA.r) * t);
    const g = Math.round(rgbA.g + (rgbB.g - rgbA.g) * t);
    const b = Math.round(rgbA.b + (rgbB.b - rgbA.b) * t);

    const toHex = (num: number) => {
      const hex = num.toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    };

    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  };

  // Determine how many regular layers exist vs special I/O layers
  // If modelData has entries whose IDs contain "projection" or "input"/"output",
  // those are I/O layers at the end. Otherwise all are regular numbered layers.
  const hasIOLayers = modelData.some(m =>
    m.id.includes('projection') || m.id.includes('input_') || m.id.includes('output_')
  );
  // Regular layer count: if I/O layers exist, the last 2 are I/O
  const regularLayerCount = hasIOLayers ? totalLayers - 2 : totalLayers;

  return (
    <div className="w-full h-full flex flex-col">
      {/* Main heatmap area */}
      <div className="flex-1 relative flex items-center">
        <div className="w-full flex items-center">
          <div className="grid gap-[1px] w-full" style={{gridTemplateColumns: `repeat(${totalLayers}, 1fr)`, height: "40px"}}>
            {Array.from({length: totalLayers}).map((_, layerIdx) => {
              const model = modelData.find((m) => m.layer === layerIdx && m.type === type);
              return (
                <div key={`${type}-${layerIdx}`}
                    className="border rounded transition-colors duration-300"
                    style={{
                      borderColor: "rgba(198, 40, 40, 0.15)",
                      backgroundColor: getValueColor(model),
                      borderRadius: "2px",
                      minWidth: "12px",
                      height: "100%"
                    }}/>
              );
            })}
          </div>
        </div>

        {/* Selection overlay */}
        <div
          className="absolute pointer-events-none z-10 rounded box-border transition-all duration-300"
          style={{
            left: `${(visibleRange.start / totalLayers) * 100}%`,
            width: `${((visibleRange.end - visibleRange.start + 1) / totalLayers) * 100}%`,
            top: 0,
            bottom: 0,
            border: '2px solid #c62828',
            backgroundColor: 'rgba(198, 40, 40, 0.08)'
          }}
        />
      </div>

      {/* Layer labels - dynamic based on model */}
      <div className="grid gap-[1px] w-full" style={{gridTemplateColumns: `repeat(${totalLayers}, 1fr)`}}>
        {Array.from({length: totalLayers}).map((_, idx) => {
          let label: string;
          if (hasIOLayers && idx === regularLayerCount) label = "I";
          else if (hasIOLayers && idx === regularLayerCount + 1) label = "O";
          else label = String(idx);
          return (
            <div key={idx}
                className="text-center text-[8px] font-medium text-gray-500 min-w-[12px]">
              {label}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default OverviewMatrix;
