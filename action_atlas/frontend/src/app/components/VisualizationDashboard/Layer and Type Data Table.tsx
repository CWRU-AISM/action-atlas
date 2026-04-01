"use client";
import React from "react";
import { ModelTableProps, ModelData } from "@/types/types";
import { Chip } from "@mui/material";
import { submitSAE } from "@/redux/features/saeSlice";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import { metricOptions } from "@/data/metricOptions";

const ModelTable: React.FC<ModelTableProps> = ({
  groupedData,
  selectedModel,
  setSelectedModel,
  setVisibleRange,
  getTopContributingAttrs,
  calculateAverageRank,
}) => {
  const tableRef = React.useRef<HTMLDivElement>(null);
  const initialRangeSetRef = React.useRef(false);
  const dispatch = useAppDispatch();
  
  // 获取选中的属性
  const { selectedAttrs } = useAppSelector((state) => state.model);

  // 获取属性的显示名称
  const getAttributeLabel = (attr: string) => {
    const option = metricOptions.find(opt => opt.value === attr);
    
    // 特殊处理 top_xx_score 属性
    if (attr.includes('top_') && attr.includes('_score')) {
      const match = attr.match(/top_(\d+)_score/);
      if (match) {
        return `Top ${match[1]}`;
      }
    }
    
    return option?.label || attr;
  };

  // 格式化显示值的函数
  const formatValue = (model: ModelData, attr: string) => {
    const value = model[attr];
    if (value && typeof value === "object" && "value" in value) {
      const numValue = (value as { value: number }).value;

      // VLA feature counts - show as integers
      if (attr.includes('_features') || attr.includes('total_')) {
        return Math.round(numValue);
      }
      // Normalized scores - show as percentage
      else if (attr.includes('_score')) {
        return (numValue * 100).toFixed(0) + '%';
      }
      else if (attr.includes('accuracy') || attr.includes('ratio') || attr.includes('variance')) {
        return (numValue * 100).toFixed(1) + '%';
      } else if (attr.includes('sparsity')) {
        return numValue.toFixed(1);
      } else {
        return numValue.toFixed(3);
      }
    }
    return "-";
  };

  const allModels = React.useMemo(() => {
    const resModels = groupedData
      .flat()
      .filter((model) => model.type === "RES");

    const recalculateRanks = (
      models: ModelData[],
      scoreKey: keyof ModelData
    ) => {
      const sortedModels = [...models].sort((a, b) => {
        const aValue =
          a[scoreKey] &&
          typeof a[scoreKey] === "object" &&
          "value" in a[scoreKey]
            ? ((a[scoreKey] as any).value as number)
            : -Infinity;
        const bValue =
          b[scoreKey] &&
          typeof b[scoreKey] === "object" &&
          "value" in b[scoreKey]
            ? ((b[scoreKey] as any).value as number)
            : -Infinity;
        return bValue - aValue; // 降序排列，值越大排名越高
      });

      sortedModels.forEach((model, index) => {
        if (
          model[scoreKey] &&
          typeof model[scoreKey] === "object" &&
          "value" in model[scoreKey]
        ) {
          const originalObj = model[scoreKey] as any;
          const newObj = { ...originalObj, rank: index + 1 };
          model[scoreKey] = newObj as any;
        }
      });
    };

    const deepCopyModels = resModels.map((model) => {
      const copy: any = { ...model };
      // Deep copy all metric objects that have value/rank structure
      Object.keys(model).forEach(key => {
        const val = model[key as keyof ModelData];
        if (val && typeof val === "object" && "value" in val) {
          copy[key] = { ...val };
        }
      });
      return copy as ModelData;
    });

    // 对选中的属性重新计算排名
    selectedAttrs.forEach(attr => {
      recalculateRanks(deepCopyModels, attr as keyof ModelData);
    });

    return deepCopyModels;
  }, [groupedData, selectedAttrs]);

  const handleTableScroll = () => {
    if (tableRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = tableRef.current;
      const totalModels = allModels.length;
      const rowHeight = scrollHeight / totalModels;
      const visibleRows = Math.ceil(clientHeight / rowHeight);
      const startIndex = Math.floor(scrollTop / rowHeight);
      const endIndex = Math.min(startIndex + visibleRows - 1, totalModels - 1);

      if (allModels.length > 0) {
        const startLayer = allModels[startIndex]?.layer || 0;
        const endLayer = allModels[endIndex]?.layer || 0;

        setVisibleRange({ start: startLayer, end: endLayer });
      } else {
        setVisibleRange({ start: 0, end: 4 });
      }
    }
  };

  React.useEffect(() => {
    if (setVisibleRange && !initialRangeSetRef.current) {
      if (allModels.length > 0) {
        const maxVisibleItems = 5; // 可见的层数
        const endIndex = Math.min(maxVisibleItems - 1, allModels.length - 1);
        const startLayer = allModels[0]?.layer || 0;
        const endLayer = allModels[endIndex]?.layer || Math.min(maxVisibleItems - 1, allModels.length - 1);
        setVisibleRange({ start: startLayer, end: endLayer });
      } else {
        setVisibleRange({ start: 0, end: 4 });
      }
      initialRangeSetRef.current = true;
    }
  }, [setVisibleRange, allModels]);

  const handleRowClick = (model: ModelData) => {
    setSelectedModel(model);
    dispatch(submitSAE(model.id));
  };

  // 计算列宽
  const columnWidth = `${100 / (selectedAttrs.length + 1)}%`;

  return (
    <div
      className="shadow-sm h-full w-full"
      style={{ display: "grid", gridTemplateRows: "auto 1fr" }}
    >
      <div className="w-full bg-gray-50">
        <table className="w-full table-fixed border-collapse">
          <thead>
            <tr className="text-red-800 text-[9px]">
              <th className="px-1 py-1 text-center font-semibold border-b" style={{ width: columnWidth }}>
                L
              </th>
              {selectedAttrs.map((attr) => (
                <th key={attr} className="px-1 py-1 text-center font-semibold border-b truncate" style={{ width: columnWidth }}>
                  {getAttributeLabel(attr).substring(0, 4)}
                </th>
              ))}
            </tr>
          </thead>
        </table>
      </div>

      <div
        ref={tableRef}
        onScroll={handleTableScroll}
        className="w-full overflow-y-auto"
      >
        <table className="w-full table-fixed border-collapse">
          <tbody>
            {allModels.map((model, index) => {
              return (
                <tr
                  key={model.id}
                  onClick={() => handleRowClick(model)}
                  className={`cursor-pointer transition-colors duration-200 text-[9px] ${
                    selectedModel?.id === model.id
                      ? "bg-red-100"
                      : index % 2 === 0
                      ? "bg-white hover:bg-red-50"
                      : "bg-gray-50 hover:bg-red-50"
                  }`}
                  style={{ height: "20px" }}
                >
                  <td className="border-b px-1 py-0.5 text-center" style={{ width: columnWidth }}>
                    {model.layer}
                  </td>

                  {selectedAttrs.map((attr) => (
                    <td key={attr} className="border-b px-1 py-0.5 text-center" style={{ width: columnWidth }}>
                      {formatValue(model, attr)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ModelTable;