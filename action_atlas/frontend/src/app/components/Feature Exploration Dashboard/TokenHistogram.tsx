"use client";
import React, { useRef, useState } from "react";
import { Point } from "@/types/types";
import { Typography } from "@mui/material";
import { Chart, registerables } from "chart.js";
import { useAppSelector } from "@/redux/hooks";

Chart.register(...registerables);

const SelectButton = ({ onClick, disabled }: { onClick: React.MouseEventHandler<HTMLButtonElement>; disabled: boolean }) => (
  <button onClick={onClick} disabled={disabled} className={`rounded text-s font-medium px-4 py-2 transition-colors ${disabled ? "bg-[#e2e8f0] text-[#94a3b8] cursor-not-allowed" : "bg-[#3182ce] text-white hover:bg-[#2c5282] hover:shadow-md"}`}>
    View Details
  </button>
);

const SentenceItem = ({ point }: { point: Point }) => {
  const normalizeThreshold = 50;
  const textColorThreshold = 0.8;

  return (
    <div key={point.id} className="border-b pb-2">
      <p className="text-gray-700">
        {point.sentence.map((item, index) => {
          const intensity = item.value > 0 ? Math.min(item.value / normalizeThreshold, 1) : 0;
          const highlightColor = `rgba(49, 130, 206, ${intensity})`;
          const textColor = intensity > textColorThreshold ? "white" : "inherit";

          return (
            <span key={index} style={{ backgroundColor: item.value > 0 ? highlightColor : "transparent", color: textColor, padding: "0 1px", borderRadius: "8px", margin: "0 1px", position: "relative", cursor: item.value > 0 ? "pointer" : "default", transition: "box-shadow 0.2s" }} className="group" title={item.value > 0 ? `Activation: ${item.value.toFixed(2)}` : ""}>
              {item.token}
              {item.value > 0 && (
                <span className="absolute invisible group-hover:visible bg-gray-800 text-white text-xs rounded px-2 py-1 left-1/2 -translate-x-1/2 bottom-full mb-1 whitespace-nowrap z-50" style={{ opacity: 0, transition: "opacity 0.2s" }} onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; }} onMouseLeave={(e) => { e.currentTarget.style.opacity = "0"; }}>
                  Activation: {item.value.toFixed(2)}
                  <span className="absolute h-2 w-2 bg-gray-800 transform rotate-45 left-1/2 -translate-x-1/2 -bottom-1"></span>
                </span>
              )}
            </span>
          );
        })}
      </p>
      <div className="mt-1 flex items-center gap-2">
        <span className="text-sm font-medium text-[#1976d2]">Keyword: {point.keyword}</span>
        <span className="text-sm text-gray-500">Similarity: {point.similarity.toFixed(3)}</span>
      </div>
    </div>
  );
};

const TokenHistogram = ({ selectedPoints }: { selectedPoints: Point[] }) => {
  const [showSentences, setShowSentences] = useState(false);
  const chartRef = useRef<HTMLDivElement>(null);
  const featureData = useAppSelector((state) => state.feature.selectedFeature?.data);

  const allPoints = React.useMemo(() => featureData?.activation_data?.map((data, index) => ({
    x: data.similarity_rank || index,
    y: data.max_value_rank || index,
    id: `point-${index}`,
    sentence: data.token_value_pairs.map((pair) => ({ token: pair.token, value: pair.value })),
    keyword: data.max_value_token,
    similarity: data.similarity,
    bin: 0,
  })) || [], [featureData]);

  const pointsToDisplay = selectedPoints.length > 0 ? selectedPoints : allPoints;

  const allTokenData = React.useMemo(() => allPoints.reduce<{[key: string]: number}>((acc, point) => ({ ...acc, [point.keyword]: (acc[point.keyword] || 0) + 1 }), {}), [allPoints]);

  const selectedTokenData = React.useMemo(() => selectedPoints.reduce<{[key: string]: number}>((acc, point) => ({ ...acc, [point.keyword]: (acc[point.keyword] || 0) + 1 }), {}), [selectedPoints]);

  const selectedKeywordsSet = React.useMemo(() => new Set(selectedPoints.map((point) => point.keyword)), [selectedPoints]);

  const combinedTokenData = React.useMemo(() => {
    const allTokens = new Set([...Object.keys(allTokenData), ...Object.keys(selectedTokenData)]);
    return Array.from(allTokens).map((token): [string, number, number] => [
      token,
      allTokenData[token] || 0,
      selectedTokenData[token] || 0
    ]).sort((a, b) => {
      const aSelected = selectedKeywordsSet.has(a[0]);
      const bSelected = selectedKeywordsSet.has(b[0]);
      if (aSelected && !bSelected) return -1;
      if (!aSelected && bSelected) return 1;
      if (aSelected && bSelected) return b[2] - a[2];
      return b[1] - a[1];
    });
  }, [allTokenData, selectedTokenData, selectedKeywordsSet]);

  const maxCount = React.useMemo(() => combinedTokenData.length ? Math.max(...combinedTokenData.map(([_, count]) => count), 1) : 1, [combinedTokenData]);

  return (
    <div className="w-full h-full flex flex-col">
      <Typography variant="subtitle1" fontWeight="bold" className="mb-2">
        {selectedPoints.length > 0 ? "Selected Tokens" : "Max Activation Tokens"}
      </Typography>
      <div className="flex flex-col gap-2 flex-grow">
        <SelectButton onClick={() => setShowSentences(true)} disabled={pointsToDisplay.length === 0} />
        <div className="flex-grow overflow-hidden h-[200px] bg-[#f7fafccc] border border-[#e2e8f0cc] rounded-lg p-2" style={{ maxHeight: "calc(100vh - 200px)" }}>
          {combinedTokenData.length > 0 ? (
            <div ref={chartRef} className="w-full h-full overflow-y-auto pr-1 max-h-[100%]" style={{ display: "block" }}>
              <div className="space-y-2">
                {combinedTokenData.map(([token, totalCount, selectedCount], index) => {
                  const isSelected = selectedKeywordsSet.has(token);
                  return (
                    <div key={index} className={`flex items-center ${isSelected ? "bg-blue-50 rounded" : ""}`}>
                      <div className={`w-16 min-w-16 text-sm font-medium truncate mr-2 ${isSelected ? "text-[#3281CE]" : "text-[#374151]"}`} title={token}>{token}</div>
                      <div className="w-[40%] h-6 bg-gray-100 rounded-sm overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-[#3182ce33] to-[rgba(49,130,206,0.1)]" style={{ width: `${(totalCount / maxCount) * 100}%` }}>
                          {selectedCount > 0 && <div className="h-full bg-gradient-to-r from-[#3182ce] to-[rgba(49,130,206,0.7)]" style={{ width: `${(selectedCount / totalCount) * 100}%` }}></div>}
                        </div>
                      </div>
                      <div className="ml-2 mr-2 text-sm text-gray-600 min-w-8 text-right">{selectedCount > 0 ? `${selectedCount}/${totalCount}` : totalCount}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <p className="text-sm text-[#718096] font-medium">No data available to display statistics</p>
          )}
        </div>
      </div>

      {showSentences && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">{selectedPoints.length > 0 ? "Selected Sentences" : "All Sentences"}</h2>
              <button onClick={() => setShowSentences(false)} className="text-gray-500 hover:text-gray-700">✕</button>
            </div>
            <div className="space-y-4">
              {pointsToDisplay.slice(0, 50).map((point) => (<SentenceItem key={point.id} point={point} />))}
              {pointsToDisplay.length > 50 && <p className="text-sm text-gray-500 italic text-center">Showing 50 of {pointsToDisplay.length} sentences</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TokenHistogram;