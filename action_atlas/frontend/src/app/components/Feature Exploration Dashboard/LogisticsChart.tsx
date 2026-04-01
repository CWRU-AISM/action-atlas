"use client";
import React, { useState } from "react";
import { FeatureSectionProps } from "@/types/types";
import { useAppSelector } from "@/redux/hooks";

const getTopData = (
  negTokens: string[],
  negValues: number[],
  posTokens: string[],
  posValues: number[]
) => {
  const mapData = (
    tokens: string[],
    values: number[],
    type: string,
    sortFn: (a: number, b: number) => number
  ) =>
    tokens
      .map((token, index) => ({ name: token, value: values[index], type }))
      .sort((a, b) => sortFn(a.value, b.value))
      .slice(0, 10);

  return {
    negativeData: mapData(negTokens, negValues, "negative", (a, b) => a - b),
    positiveData: mapData(posTokens, posValues, "positive", (a, b) => b - a),
  };
};

const LogisticsChart = ({ featureInfo }: FeatureSectionProps) => {
  const [hoveredItem, setHoveredItem] = useState<{
    index: number;
    side: "positive" | "negative";
  } | null>(null);
  const selectedFeature = useAppSelector(
    (state) => state.feature.selectedFeature
  );
  const displayInfo = featureInfo || selectedFeature;

  const { negativeData, positiveData } = getTopData(
    displayInfo?.data.raw_stats.neg_tokens.tokens || [],
    displayInfo?.data.raw_stats.neg_tokens.values || [],
    displayInfo?.data.raw_stats.pos_tokens.tokens || [],
    displayInfo?.data.raw_stats.pos_tokens.values || []
  );

  const smallestNegValue = Math.abs(negativeData[0]?.value || 1) || 1;
  const largestPosValue = positiveData[0]?.value || 1;

  const renderBar = (
    item: { name: string; value: number },
    index: number,
    side: "positive" | "negative",
    width: number,
    gradient: string
  ) => {
    const isHovered =
      hoveredItem?.index === index && hoveredItem?.side === side;
    return (
      <div
        key={`${side}-${index}`}
        className={`relative w-full flex ${
          side === "negative" ? "justify-end" : "justify-start"
        }`}
      >
        <div
          className={`h-7 flex items-center ${
            side === "negative" ? "justify-end rounded-l-md" : "rounded-r-md"
          } transition-all duration-300`}
          style={{
            width: `${width}%`,
            background: gradient,
            opacity: isHovered ? 0.8 : 1,
          }}
          onMouseEnter={() => setHoveredItem({ index, side })}
          onMouseLeave={() => setHoveredItem(null)}
        >
          <span className="text-[17px] text-white px-2 truncate font-bold font-mono">
            {item.name}
          </span>
          {isHovered && (
            <div className="absolute left-0 top-[-24px] bg-gray-800 text-white text-xs py-1 px-2 rounded shadow-lg z-10">
              Value: {Math.abs(item.value).toFixed(4)}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full">
      <div className="mb-2">
        <h2 className="text-[18px] font-bold text-center">Vocabulary Space</h2>
      </div>
      <div className="flex justify-center mb-2 gap-6">
        <div className="flex items-center">
          <div
            className="w-4 h-4 rounded-sm mr-2"
            style={{ background: "rgba(128, 128, 128, 0.6)" }}
          ></div>
          <span className="text-sm">Negative Logits</span>
        </div>
        <div className="flex items-center">
          <div
            className="w-4 h-4 rounded-sm mr-2"
            style={{ background: "rgba(25, 118, 210, 0.8)" }}
          ></div>
          <span className="text-sm">Positive Logits</span>
        </div>
      </div>
      <div className="flex h-full w-full justify-center mx-auto">
        <div className="w-[45%] flex flex-col items-end gap-2">
          {negativeData.map((item, index) =>
            renderBar(
              item,
              index,
              "negative",
              (Math.abs(item.value) / smallestNegValue) * 100,
              "linear-gradient(to left, rgba(128, 128, 128, 0.6), rgba(128, 128, 128, 0.4))"
            )
          )}
        </div>
        <div className="w-[10%] flex flex-col items-center">
          {negativeData.map((_, index) => (
            <div
              key={`divider-${index}`}
              className="h-9 w-full flex justify-center items-center"
            >
              <div className="w-[2px] h-full bg-gray-300"></div>
            </div>
          ))}
        </div>
        <div className="w-[45%] flex flex-col items-start gap-2">
          {positiveData.map((item, index) =>
            renderBar(
              item,
              index,
              "positive",
              (item.value / largestPosValue) * 100,
              "linear-gradient(to right, rgba(25, 118, 210, 0.8), rgba(25, 118, 210, 0.5))"
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default LogisticsChart;
