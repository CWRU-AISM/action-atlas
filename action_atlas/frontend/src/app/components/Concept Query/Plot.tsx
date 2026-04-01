"use client";
import React, { useEffect } from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  LogarithmicScale,
  TooltipItem,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { PlotProps } from "@/types/types";
import { useAppSelector } from "@/redux/hooks";

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const applyChartGradient = (chart: ChartJS<"line">) => {
  const ctx = chart.ctx;
  chart.data.datasets.forEach((dataset, index) => {
    const gradient = ctx.createLinearGradient(0, 0, 0, chart.height);
    gradient.addColorStop(
      0,
      index === 0 ? "rgba(54, 162, 235, 0.3)" : "rgba(255, 99, 71, 0.3)"
    );
    gradient.addColorStop(
      1,
      index === 0 ? "rgba(54, 162, 235, 0)" : "rgba(255, 99, 71, 0)"
    );
    dataset.backgroundColor = gradient;
  });
  chart.update();
};

const Plot: React.FC<PlotProps> = ({ queryResult, chartRef }) => {
  const optimizedQueryResult = useAppSelector(
    (state) => state.query.optimizedQueryResult
  );

  useEffect(() => {
    if (chartRef.current) applyChartGradient(chartRef.current);
  }, [chartRef, queryResult, optimizedQueryResult]);

  const createCombinedData = () => {
    const labels = queryResult.bins.slice(0, -1).map((bin) => (bin ?? 0).toFixed(2));

    console.log("bins", queryResult.bins.slice(0, -1));

    // 预处理函数：将孤立点设置为 null
    const preprocessData = (data: (number | null)[]) => {
      // 第一步：将 0 和 null 转换为 0.1
      const processedData = data.map((value) => (value === 0 ? 0.1 : value));

      // 第二步：处理孤立点
      const result = processedData.map((value, index) => {
        if (value === 0.1) {
          const prevValue = index > 0 ? processedData[index - 1] : null;
          const nextValue =
            index < processedData.length - 1 ? processedData[index + 1] : null;

          // 如果前后都是 0.1，则将当前值设为 null
          if (prevValue === 0.1 && nextValue === 0.1) {
            return null;
          } else if (prevValue !== 0.1 && nextValue !== 0.1) {
            return null;
          }
        }
        return value;
      });

      // 第三步：从前往后扫描，找到第一个0.1的值的位置
      let firstNonZeroIndex = 0;
      if (result[0] === 0.1) {
        firstNonZeroIndex = 1;
        while (
          firstNonZeroIndex < result.length &&
          result[firstNonZeroIndex] !== 0.1
        ) {
          firstNonZeroIndex++;
        }
      }

      // 从后往前扫描，找到第一个0.1的值的位置
      let lastNonZeroIndex = result.length - 1;
      if (result[lastNonZeroIndex] === 0.1) {
        lastNonZeroIndex = result.length - 2;
        while (lastNonZeroIndex >= 0 && result[lastNonZeroIndex] !== 0.1) {
          lastNonZeroIndex--;
        }
      }

      // 将中间所有的0.1都变成null
      return result.map((value, index) => {
        if (
          value === 0.1 &&
          index > firstNonZeroIndex &&
          index < lastNonZeroIndex
        ) {
          return null;
        }
        return value;
      });
    };

    const datasets = [
      {
        label: "Original Query",
        data: preprocessData(queryResult.counts),
        borderColor: "rgb(54, 162, 235)",
        borderWidth: 2,
        fill: true,
      },
    ];

    console.log(preprocessData(queryResult.counts));

    if (optimizedQueryResult) {
      datasets.push({
        label: "Optimized Query",
        data: preprocessData(optimizedQueryResult.counts),
        borderColor: "rgb(255, 99, 71)",
        borderWidth: 2,
        fill: true,
      });
    }

    return { labels, datasets };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { reverse: true, title: { display: true, text: "Similarity" } },
      y: {
        type: "logarithmic" as const,
        title: { display: true, text: "Count" },
        ticks: {
          // 自定义刻度，确保不显示0.1
          callback: function(tickValue: any) {
            // 避免显示接近0.1的刻度值
            if (tickValue <= 0.1 || (tickValue > 0.1 && tickValue < 0.2)) {
              return '0';
            }
            return tickValue;
          }
        }
      },
    },
    spanGaps: true,
    plugins: {
      legend: { display: true, position: "top" as const },
      tooltip: {
        callbacks: {
          title: (context: TooltipItem<"line">[]) =>
            `Similarity: ${context[0].label}`,
        },
      },
    },
  };

  return (
    <Line ref={chartRef} data={createCombinedData()} options={chartOptions} />
  );
};

export default Plot;
