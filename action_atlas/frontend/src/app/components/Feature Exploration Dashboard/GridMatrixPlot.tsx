"use client";
import React from "react";
import { useAppSelector } from "@/redux/hooks";
import {
  Coordinates,
  MatrixPlotProps,
  Point,
  SelectionBox,
} from "@/types/types";
import { throttle } from "lodash";

const INTERVAL_COLORS = [
  "#D1884B",
  "#D1B85A",
  "#9DD155",
  "#50D180",
  "#42CCC6",
  "#3281CE",
];
const CROSS_COLORS = [
  "#B37440",
  "#B39C4D",
  "#87B349",
  "#44B36C",
  "#39B3AC",
  "#2B6FB3",
];

const GridMatrixPlot = ({ onPointsSelected }: MatrixPlotProps) => {
  const plotRef = React.useRef<HTMLDivElement | null>(null);
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const [points, setPoints] = React.useState<Point[]>([]);
  const [selectionBox, setSelectionBox] = React.useState<SelectionBox | null>(
    null
  );
  const [selectedPoints, setSelectedPoints] = React.useState<Point[]>([]);
  const [dragging, setDragging] = React.useState(false);
  const [startCoords, setStartCoords] = React.useState<Coordinates | null>(
    null
  );
  const [plotSize, setPlotSize] = React.useState({ width: 10, height: 10 });

  const [binsData, setBinsData] = React.useState<
    Array<{
      bin_range: [number, number];
      sample_count: number;
      bin_contains: string;
    }>
  >([]);
  const [totalSamples, setTotalSamples] = React.useState(0);

  const featureData = useAppSelector(
    (state) => state.feature.selectedFeature?.data
  );

  const binsStatistics = useAppSelector(
    (state) => state.feature.selectedFeature?.data?.bins_statistics
  );

  React.useEffect(() => {
    if (binsStatistics?.bins_data) {
      const sortedBinsData = [...binsStatistics.bins_data].sort((a, b) => {
        if (a.bin_range[0] === -1) return 1;
        if (b.bin_range[0] === -1) return -1;
        return a.bin_range[0] - b.bin_range[0];
      });

      setBinsData(sortedBinsData);
      setTotalSamples(binsStatistics.total_samples || 0);
      console.log("Bins statistics loaded:", sortedBinsData);
    } else {
      console.log("No bins_statistics found");
      setBinsData([]);
      setTotalSamples(0);
    }
  }, [binsStatistics]);

  const constructTokensArray = React.useCallback(
    (
      tokenValuePairs: { token: string; value: number }[]
    ): {
      token: string;
      value: number;
    }[] => {
      return tokenValuePairs
        .map((pair) => ({
          token: pair.token.replace(/^##/, ""), // 处理子词标记（去掉 ##）
          value: pair.value,
        }))
        .filter((pair) => !pair.token.match(/^\[.*]$/)) // 移除特殊标记，如 [CLS], [SEP] 等
        .filter((pair) => pair.token.trim().length > 0); // 过滤掉空字符串
    },
    []
  );

  const generatePoints = React.useCallback((): Point[] => {
    if (!featureData?.activation_data) {
      console.log("No activation data found");
      return [];
    }

    const activationData = featureData.activation_data;

    return activationData.map((data, index) => {
      return {
        x: data.similarity_rank || index,
        y: data.max_value_rank || index,
        id: `point-${index}`,
        sentence: constructTokensArray(data.token_value_pairs),
        keyword: data.max_value_token,
        similarity: data.similarity,
        bin: 0, // 将在渲染时确定
        intervalBin: data.Interval_Bin,
        intervalContains: data.Interval_contains,
      };
    });
  }, [featureData, constructTokensArray]);

  React.useEffect(() => {
    const newPoints = generatePoints();
    setPoints(newPoints);
    setSelectedPoints([]);
    setSelectionBox({ left: 0, top: 0, width: 0, height: 0 });
  }, [featureData, generatePoints]);

  React.useEffect(() => {
    if (containerRef.current) {
      const { offsetWidth, offsetHeight } = containerRef.current;
      const minSize = Math.min(offsetWidth, offsetHeight) * 0.85;
      setPlotSize({ width: minSize, height: minSize });
    }
  }, []);

  const getRelativeCoordinates = (e: React.MouseEvent): Coordinates => {
    const rect = plotRef.current?.getBoundingClientRect();
    return rect
      ? { x: e.clientX - rect.left, y: e.clientY - rect.top }
      : { x: 0, y: 0 };
  };

  const handleMouse = (e: React.MouseEvent | null, isDown: boolean) => {
    if (isDown && e) {
      const coords = getRelativeCoordinates(e);

      if (plotRef.current) {
        const plotRect = plotRef.current.getBoundingClientRect();
        if (
          coords.x < 0 ||
          coords.y < 0 ||
          coords.x > plotRect.width ||
          coords.y > plotRect.height
        ) {
          return; // 如果点击在网格外，不启动选择
        }
      }

      setStartCoords(coords);
      setDragging(true);
      setSelectionBox({ left: coords.x, top: coords.y, width: 0, height: 0 });
      setSelectedPoints([]);
    } else {
      setDragging(false);
      setStartCoords(null);
    }
  };

  const handleMouseMove = throttle((e: React.MouseEvent) => {
    if (!dragging || !startCoords || !plotRef.current) return;
    const coords = getRelativeCoordinates(e);
    const plotRect = plotRef.current.getBoundingClientRect();
    const boundedX = Math.max(0, Math.min(plotRect.width, coords.x));
    const boundedY = Math.max(0, Math.min(plotRect.height, coords.y));
    const width = boundedX - startCoords.x;
    const height = boundedY - startCoords.y;
    const box = {
      left: width > 0 ? startCoords.x : boundedX,
      top: height > 0 ? startCoords.y : boundedY,
      width: Math.abs(width),
      height: Math.abs(height),
    };

    setSelectionBox(box);

    if (plotRef.current) {
      const plotWidth = plotRect.width;
      const plotHeight = plotRect.height;

      const pointRadius = 4;

      setSelectedPoints(
        points.filter((point) => {
          const safeTotalSamples = totalSamples || 1;
          let x = ((safeTotalSamples - point.x) / safeTotalSamples) * plotWidth;
          let y = (point.y / safeTotalSamples) * plotHeight;

          x = Math.max(pointRadius, Math.min(plotWidth - pointRadius, x));
          y = Math.max(pointRadius, Math.min(plotHeight - pointRadius, y));

          const pointLeft = x - pointRadius;
          const pointRight = x + pointRadius;
          const pointTop = y - pointRadius;
          const pointBottom = y + pointRadius;

          return (
            box.left + box.width >= pointLeft &&
            box.left <= pointRight &&
            box.top + box.height >= pointTop &&
            box.top <= pointBottom
          );
        })
      );
    }
  }, 16);

  React.useEffect(() => {
    onPointsSelected(selectedPoints);
  }, [selectedPoints, onPointsSelected]);

  const renderGridCells = () => {
    if (!plotRef.current || binsData.length === 0) return null;

    const cells = [];
    const totalCount =
      totalSamples || binsData.reduce((sum, bin) => sum + bin.sample_count, 0) || 1;

    const binsDataForDisplay = [...binsData];

    const cumulativeCounts = [];
    const yPositions = [];
    let cumulativeCount = 0;
    let cumulativeHeight = 0;

    for (let i = 0; i < binsDataForDisplay.length; i++) {
      cumulativeCount += binsDataForDisplay[i].sample_count;
      cumulativeCounts.push(cumulativeCount);

      const heightPercent =
        (binsDataForDisplay[i].sample_count / totalCount) * 100;
      yPositions.push(100 - cumulativeHeight - heightPercent); // 从底部向上计算位置
      cumulativeHeight += heightPercent;
    }

    let currentX = 0;

    for (let i = 0; i < binsDataForDisplay.length; i++) {
      const bin = binsDataForDisplay[i];
      const heightPercent = (bin.sample_count / totalCount) * 100;
      const widthPercent = (bin.sample_count / totalCount) * 100;
      const currentY = yPositions[i]; // 使用预计算的Y位置
      const bgColor = INTERVAL_COLORS[i];
      const crossColor = CROSS_COLORS[i];

      cells.push(
        <div
          key={`cell-x-${i}`}
          className="absolute border-0"
          style={{
            left: `${currentX}%`,
            top: 0,
            width: `1px`,
            height: "100%",
            backgroundColor: "white",
            zIndex: 5,
          }}
        ></div>
      );
      cells.push(
        <div
          key={`cell-y-${i}`}
          className="absolute border-0"
          style={{
            left: 0,
            top: `${currentY}%`,
            width: "100%",
            height: `1px`,
            backgroundColor: "white",
            zIndex: 5,
          }}
        >
          <div className="absolute -left-[220px] transform -translate-y-1/2 text-x text-gray-600 w-[210px] text-right">
            {bin.bin_range[0] === -1 ? (
              // 旋转容器但保持定位不变
              <div className="transform origin-right -translate-x-2 translate-y-5 -rotate-90">
                Top Feature
              </div>
            ) : (
              <div>{bin.bin_range[1].toFixed()}</div>
            )}
          </div>
        </div>
      );

      cells.push(
        <div
          key={`cell-bg-row-${i}`}
          className="absolute"
          style={{
            left: 0,
            top: `${currentY}%`,
            width: "100%",
            height: `${heightPercent}%`,
            backgroundColor: bgColor,
            opacity: 0.1, // 降低不透明度，使行背景更淡
          }}
        />
      );

      cells.push(
        <div
          key={`cell-bg-cross-${i}`}
          className="absolute"
          style={{
            left: `${currentX}%`,
            top: `${currentY}%`,
            width: `${widthPercent}%`,
            height: `${heightPercent}%`,
            backgroundColor: crossColor,
            opacity: 0.1, // 对角线单元格不透明度更高
          }}
        />
      );

      currentX += widthPercent;
    }

    cells.push(
      <div
        key="x-axis-title"
        className="absolute text-x text-gray-700 font-bold"
        style={{
          bottom: "-40px",
          left: "50%",
          transform: "translateX(-50%)",
          width: "100%",
          textAlign: "center",
        }}
      >
        Similarity Rank
      </div>
    );

    cells.push(
      <div
        key="y-axis-title"
        className="absolute text-x text-gray-600 font-bold"
        style={{
          top: "-25px",
          left: "-50px",
          transform: "rotate(0deg)",
          width: "100px",
          textAlign: "center",
        }}
      >
        Activation
      </div>
    );

    const xAxisPositions: number[] = [];
    let accumulatedPercent = 0;

    for (let i = 0; i < binsDataForDisplay.length; i++) {
      const bin = binsDataForDisplay[i];
      const widthPercent = (bin.sample_count / totalCount) * 100;
      xAxisPositions.push(accumulatedPercent + widthPercent);
      accumulatedPercent += widthPercent;
    }

    cumulativeCounts.forEach((count, index) => {
      cells.push(
        <div
          key={`x-axis-value-${index}`}
          className="absolute text-x text-gray-600"
          style={{
            bottom: "-20px",
            left: `${xAxisPositions[index]}%`,
            transform: "translateX(-50%)",
            whiteSpace: "nowrap",
          }}
        >
          {`${totalCount - count + 1}`}
        </div>
      );
    });

    return cells;
  };

  const renderDataPoints = () => {
    if (!plotRef.current || points.length === 0 || totalSamples === 0)
      return null;
    const plotRect = plotRef.current.getBoundingClientRect();
    const plotWidth = plotRect.width;
    const plotHeight = plotRect.height;

    return points.map((point, index) => {
      let x =
        ((totalSamples - point.x) / totalSamples) * plotWidth +
        210 / totalSamples;
      let y = (point.y / totalSamples) * plotHeight - 270 / totalSamples;
      const isSelected = selectedPoints.some((p) => p.id === point.id);
      const diff = Math.abs(point.x - point.y);
      let diffRatio = diff / totalSamples;
      const r = Math.round(25 + (200 - 25) * diffRatio); // 从6B到A9
      const g = Math.round(118 + (200 - 118) * diffRatio); // 从8A到A9
      const b = Math.round(240 + (200 - 240) * diffRatio); // 从F7到A9
      const color = isSelected ? "#EAB308" : `rgb(${r}, ${g}, ${b})`;
      return (
        <div
          key={point.id}
          className="absolute cursor-pointer z-20" // 提高z-index以确保点显示在网格线上方
          style={{
            left: `${x}px`,
            top: `${y}px`,
            transform: "translate(-50%, -50%)",
          }}
          title={`${point.keyword}: Rank ${point.y}, Similarity Rank: ${point.x}, Diff: ${diff}`}
        >
          <div
            className="transition-all duration-200"
            style={{
              width: `${400 / totalSamples}px`,
              height: `${400 / totalSamples}px`,
              backgroundColor: color,
              opacity: isSelected ? 0.9 : 0.7,
            }}
          />
        </div>
      );
    });
  };

  return (
    <div
      ref={containerRef}
      className="h-full w-full flex flex-col items-center justify-center relative"
    >
      <div className="w-full flex justify-between items-center mb-2 px-10">
        {/* 左侧空白区域，用于平衡布局
        <div className="invisible">占位</div> */}
        
        {/* 中间标题 */}
        <h2 className="text-[18px] font-bold ml-40">Text Segment Samples</h2>

        {/* 右上角 - 点颜色图注 */}
        <div className="bg-white/90 rounded px-1 py-0.3 flex flex-col items-center border border-gray-200 shadow-sm">
          <div
            className="w-28 h-1 rounded-sm mb-2"
            style={{
              background:
                "linear-gradient(to right, rgb(25, 118, 210), rgb(200, 200, 200))",
            }}
          />
          <div className="w-full flex justify-between text-xs">
            <span className="text-[#1976d2] font-semibold">Match</span>
            <span className="text-[#686868] font-medium">Dismatch</span>
          </div>
        </div>
      </div>

      <div className="relative flex">
        <div className="flex flex-col">
        {/* 右侧激活值图注 - 垂直长条，离散颜色 */}
        <div className="absolute right-[-35px] top-[48%] transform -translate-y-1/2 bg-white/60 rounded px-0.5 py-1 flex flex-col items-center border border-gray-200 shadow-sm z-10">
          <div className="text-xs font-medium mb-2 text-gray-700">Act.</div>
          <div className="flex flex-col items-center text-xs space-y-2">
            <span className="text-[#3281CE] font-semibold">High</span>
          </div>
          <div className="w-1 h-96 rounded-sm mb-2 flex flex-col">
            <div className="flex-1" style={{ backgroundColor: '#3281CE', opacity: 0.5 }}></div>
            <div className="flex-1" style={{ backgroundColor: '#42CCC6', opacity: 0.5 }}></div>
            <div className="flex-1" style={{ backgroundColor: '#50D180', opacity: 0.5 }}></div>
            <div className="flex-1" style={{ backgroundColor: '#9DD155', opacity: 0.5 }}></div>
            <div className="flex-1" style={{ backgroundColor: '#D1B85A', opacity: 0.5 }}></div>
            <div className="flex-1" style={{ backgroundColor: '#D1884B', opacity: 0.5 }}></div>
          </div>
          <div className="flex flex-col items-center text-xs space-y-2">
            <span className="text-[#D1884B] font-medium">Low</span>
          </div>
        </div>
          
          <div
            ref={plotRef}
            className="relative border border-gray-300 select-none"
            style={{
              width: plotSize.width,
              height: plotSize.height,
              marginLeft: "20px",
              marginBottom: "30px",
            }}
            onMouseDown={(e) => handleMouse(e, true)}
            onMouseMove={handleMouseMove}
            onMouseUp={() => handleMouse(null, false)}
            onMouseLeave={() => handleMouse(null, false)}
          >
            {renderGridCells()}
            {renderDataPoints()}
            {selectionBox && (
              <div
                className="absolute border-2 border-[#1976d2] bg-[#1976d2] bg-opacity-10 z-20"
                style={{
                  left: `${selectionBox.left}px`,
                  top: `${selectionBox.top}px`,
                  width: `${selectionBox.width}px`,
                  height: `${selectionBox.height}px`,
                  pointerEvents: "none",
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GridMatrixPlot;
